"""Continuation arm configs + launcher generation for base_retention_msa_probe.

Four STATEFUL arms, all resuming from the SAME six 5B final checkpoints
(resolved from the CCRS checkpoint manifest, never guessed) WITH the source
run's full stall state (state_bootstrap: restored curriculum_control gate
state, rebuilt observer memory, env clock resume), matched seeds, and a
matched worker-local continuation budget of 6 committed ticks (480M frames
at the REAL stage-3 cadence of 80M/tick):

- BASE_STATEFUL         : the original base-retention flags, unchanged —
                          the matched "still stuck" control.
- GATE_RELAX_STATEFUL   : ONLY late_floor_retention_min_retention
                          0.88 -> 0.75 (the pre-existing hard minimum);
                          everything else identical.
- MSA_STATEFUL          : original numeric gates unchanged; the MSA probe
                          sidecar runs the tau_plus microdose on the SAME
                          restored state.
- MSA_PERMUTED_STATEFUL : MSA with the pre-registered permuted second-half
                          routing (negative control for exposure-only
                          gains).

Every arm launcher runs the state-faithful bootstrap BEFORE any worker
starts and refuses to launch if the bootstrap or its fingerprint fails; the
continuation guard then enforces the stall reproduction gate on the first
committed interval (>= 4/6 workers must hold their source tolerance) before
any MSA cycle may begin.

This module only WRITES launcher scripts + configs (chmod +x); it never
executes them. Probe frames are charged to each worker's monotonic ledger
because the probe trains through the normal supervisor loop under a forced
tolerance — MAX_ITER already includes them (no free extra training).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from dexpbt_isaaclab.msa.decision import (
    BOOTSTRAP_REPEATS,
    BOOTSTRAP_SEED,
    PERMUTED_ROUTE_SEED,
    TOP_K_RECIPIENTS,
)
from dexpbt_isaaclab.msa.controller import PROBE_PACKET_FRAMES
from dexpbt_isaaclab.msa.continuation_guard import (
    COMMITTED_TICK_FRAMES,
    TICK_SLACK_FRAMES,
)
from dexpbt_isaaclab.msa.eval_bridge import (
    FORMAL_NUM_SEEDS,
    PILOT_NUM_SEEDS,
    SMOKE_NUM_SEEDS,
    build_seed_manifest,
    eval_cost_estimate,
)
from dexpbt_isaaclab.msa.provenance import EXPOSURE_WINDOW_FRAMES
from dexpbt_isaaclab.msa.state_bootstrap import env_clock_from_snapshot

ARM_NAMES = ("BASE_STATEFUL", "GATE_RELAX_STATEFUL", "MSA_STATEFUL",
             "MSA_PERMUTED_STATEFUL")
MSA_ARM_NAMES = ("MSA_STATEFUL", "MSA_PERMUTED_STATEFUL")

# Worker-local continuation budget at the REAL committed cadence.
# 80M/tick (stage-3 SELECTION: base 20M pbt_interval x 4.0 multiplier —
# measured in the source run and confirmed by the 2026-07-19 smoke; the
# base 20M is NOT the committed quantum). Tick budget:
#   tick 1   readiness + stall REPRODUCTION interval,
#   ticks 2-3 the MSA probe cycle (common packet + second packet),
#   ticks 4-5 post-cycle same-tau observation,
#   tick 6   slack for hold/eval phase quantization.
CONTINUATION_TICKS = 6
CONTINUATION_FRAMES = CONTINUATION_TICKS * COMMITTED_TICK_FRAMES
# The strict-5B run reached 5B population frames in 44500 RL epochs
# (~112,360 frames/epoch at 8192 envs); +480M worker-local frames ~ +4280.
BASE_MAX_ITER = 44_500
CONTINUATION_MAX_ITER = BASE_MAX_ITER + 4_280

ORIGINAL_LAUNCHER = (
    "results/launchers/start_regrasping_cser_dexpbt_conservative_retention_strict_5b.sh"
)

# The single GATE_RELAX flag change (pre-existing hard minimum, NOT a new
# relaxation invented for this experiment).
GATE_RELAX_FLAG = "--late_floor_retention_min_retention"
GATE_RELAX_OLD = "0.88"
GATE_RELAX_NEW = "0.75"

# Profile expansion is AUTHORITATIVE for non-legacy profiles: a
# profile-controlled flag set on the CLI is silently reverted to the table
# value at startup. late_floor_retention_min_retention IS profile-controlled
# (in _CSER_DEXPBT_CONSERVATIVE), so GATE_RELAX must select a dedicated
# profile carrying the 0.75 table value; the CLI flag alone would be a no-op.
BASE_PROFILE = "cser_dexpbt_conservative"
GATE_RELAX_PROFILE = "cser_dexpbt_conservative_gate_relax"

# The base-retention same-tau retention threshold consumed by the MSA
# promotion gate (rho >= this). Reuses the CURRENT applicable original
# threshold; MSA does not invent a new number.
MSA_RHO_THRESHOLD = 0.88


@dataclass
class TipCheckpoint:
    worker_id: int
    iteration: int
    frame: int
    checkpoint_path: str
    live_tolerance: float
    lineage_id: str
    sha256_head: str = ""


def resolve_tip_checkpoints(manifest_csv: Path,
                            num_policies: int = 6) -> list[TipCheckpoint]:
    """Resolve the six 5B tip checkpoints from the CCRS manifest.

    Fails closed: every worker 0..num_policies-1 must have exactly one row
    whose roles include 'tip' and whose file exists on disk. The bootstrap
    hash (size + first-4MB sha256, same scheme as the CCRS manifest) is
    computed here and re-verified fail-closed by continuation_guard
    --verify-only before any arm may launch.
    """
    from dexpbt_isaaclab.msa.continuation_guard import ckpt_head_hash

    by_worker: dict[int, list[dict]] = {}
    with open(manifest_csv, newline="") as f:
        for row in csv.DictReader(f):
            if "tip" in (row.get("roles") or ""):
                by_worker.setdefault(int(row["worker_id"]), []).append(row)
    tips: list[TipCheckpoint] = []
    for w in range(num_policies):
        rows = by_worker.get(w, [])
        if len(rows) != 1:
            raise ValueError(
                f"worker {w}: expected exactly 1 tip row in manifest, "
                f"got {len(rows)}")
        r = rows[0]
        path = Path(r["checkpoint_path"])
        if not path.is_file():
            raise FileNotFoundError(f"tip checkpoint missing: {path}")
        tips.append(TipCheckpoint(
            worker_id=w, iteration=int(r["iteration"]), frame=int(r["frame"]),
            checkpoint_path=str(path),
            live_tolerance=float(r["live_tolerance"]),
            lineage_id=r.get("lineage_id") or "",
            sha256_head=ckpt_head_hash(path)))
    return tips


def arm_config(arm: str, tips: list[TipCheckpoint], source_run: str,
               base_seed: int = 42) -> dict:
    """Machine-readable config for one arm (written next to its launcher)."""
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown arm {arm}")
    manifest_id, seeds = build_seed_manifest(PILOT_NUM_SEEDS)
    wm_profile = (GATE_RELAX_PROFILE if arm == "GATE_RELAX_STATEFUL"
                  else BASE_PROFILE)
    cfg: dict = {
        "arm": arm,
        "profile": "base_retention_msa_probe",
        "wm_strategy_profile": wm_profile,
        "state_faithful": True,
        "source_run": str(source_run),
        "resume_checkpoints": {
            f"p{t.worker_id}": {
                "path": t.checkpoint_path,
                "iteration": t.iteration,
                "frame": t.frame,
                "live_tolerance": t.live_tolerance,
                "lineage_id": t.lineage_id,
                "sha256_head": t.sha256_head,
            } for t in tips
        },
        "base_seed": base_seed,           # matched across all arms
        "committed_tick_frames": COMMITTED_TICK_FRAMES,
        "continuation_ticks_per_worker": CONTINUATION_TICKS,
        "continuation_frames_per_worker": CONTINUATION_FRAMES,
        "cap_slack_frames": TICK_SLACK_FRAMES,
        "budget_breakdown_ticks": {
            "readiness_and_stall_reproduction": 1,
            "msa_probe_cycle_common_plus_second": 2,
            "post_cycle_same_tau_observation": 2,
            "phase_quantization_slack": 1,
        },
        "max_iterations": CONTINUATION_MAX_ITER,
        "frame_ledger": {
            "monotonic": True,
            "probe_frames_charged": True,
            "note": "probe packets train through the normal supervisor loop "
                    "under forced tolerance; they consume the same 480M "
                    "budget as ordinary continuation frames",
        },
        "original_launcher": ORIGINAL_LAUNCHER,
        "flag_diff_vs_base": [],
    }
    if arm == "GATE_RELAX_STATEFUL":
        cfg["flag_diff_vs_base"] = [
            {"flag": GATE_RELAX_FLAG, "old": GATE_RELAX_OLD,
             "new": GATE_RELAX_NEW,
             "note": "already the configured hard minimum "
                     "(late_floor_retention_hard_min_retention); applied via "
                     f"the dedicated {GATE_RELAX_PROFILE} profile table "
                     "because profile expansion is authoritative over CLI"}]
    if arm in MSA_ARM_NAMES:
        cfg["msa"] = {
            "tau_plus_rule": "max(0.010, 0.95 * tau[w]) per worker "
                             "(heterogeneous, criterion-local)",
            "exposure_window_frames": EXPOSURE_WINDOW_FRAMES,
            "packet_frames": PROBE_PACKET_FRAMES,
            "packet_note": "one committed PBT tick at the REAL stage-3 "
                           "cadence (80M env frames; the base 20M "
                           "pbt_interval is NOT the committed quantum); "
                           "provenance stores OBSERVED committed frames",
            "top_k_recipients": TOP_K_RECIPIENTS,
            "permuted_route": arm == "MSA_PERMUTED_STATEFUL",
            "permuted_route_seed": PERMUTED_ROUTE_SEED,
            "eval_num_seeds_smoke": SMOKE_NUM_SEEDS,
            "eval_num_seeds_pilot": PILOT_NUM_SEEDS,
            "eval_num_seeds_formal": FORMAL_NUM_SEEDS,
            "eval_num_seeds_note": "N=4 execution smoke is the launcher "
                                   "default (machinery validation only, "
                                   "never a method conclusion); pilot N=16 "
                                   "must be set explicitly (MSA_NUM_SEEDS); "
                                   "formal N=256 is formal-only and not "
                                   "affordable on 2 eval GPUs without "
                                   "further optimization",
            "eval_seed_manifest_id_pilot": manifest_id,
            "eval_seed_first": seeds[0],
            "eval_seed_last": seeds[-1],
            "eval_cells_per_cycle": {"pre": 12, "mid": 6, "post": 9,
                                     "total": 27},
            "eval_cost_estimates": [
                eval_cost_estimate(SMOKE_NUM_SEEDS, 2),
                eval_cost_estimate(PILOT_NUM_SEEDS, 2),
                eval_cost_estimate(FORMAL_NUM_SEEDS, 2),
            ],
            "eval_device_note": "sidecar --eval-devices is REQUIRED; the "
                                "launcher pins ${MSA_EVAL_DEVICES:-"
                                "cuda:6,cuda:7} (GPUs outside DEXPBT_GPUS), "
                                "never the trainers' cuda:0; at most one "
                                "Isaac eval process per device",
            "bootstrap": {"seed": BOOTSTRAP_SEED, "repeats": BOOTSTRAP_REPEATS,
                          "quantile": 0.05,
                          "statistic": "criterion-local conditional panel "
                                       "median of per-worker L_mid"},
            "rho_threshold": MSA_RHO_THRESHOLD,
            "rho_rule": "p_post/p_pre at the worker's OWN tau; p_pre>=0.01 "
                        "required; cross-tolerance historical peaks "
                        "forbidden",
            "numeric_gates": "unchanged from base-retention; only the "
                             "cross-tolerance retention read is replaced by "
                             "same-tau rho",
        }
    return cfg


_LAUNCHER_TEMPLATE = """#!/usr/bin/env bash
# {arm} continuation arm of base_retention_msa_probe (STATE-FAITHFUL).
# GENERATED by dexpbt_isaaclab.msa.arms — DO NOT LAUNCH without the
# LAUNCH_CHECKLIST sign-off. This arm resumes all six workers from the 5B
# tip checkpoints (per-policy warm start) PLUS the source run's full stall
# state (curriculum_control gate state, rebuilt observer memory, env clock)
# with matched seeds.
set -euo pipefail

if [[ "${{1:-}}" != "--i-know-what-i-am-doing" ]]; then
  echo "REFUSING TO LAUNCH: pass --i-know-what-i-am-doing after completing"
  echo "results/analysis/msa_probe_implementation/LAUNCH_CHECKLIST.md"
  exit 2
fi

RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${{RESULTS_ROOT}}/runs/base_retention_msa_probe_{arm_lower}_${{DATE}}"
LOG="${{RUN_DIR}}/launch_master.log"
BASE_SEED="${{BASE_SEED:-42}}"
SOURCE_RUN="{source_run}"

RECOVER_LIST="{recover_list}"
TOLERANCE_LIST="{tolerance_list}"
CLOCK_LIST="{clock_list}"

export DEXPBT_MAX_ACTIVE="${{DEXPBT_MAX_ACTIVE:-6}}"
export DEXPBT_INTER_LAUNCH_DELAY_S="${{DEXPBT_INTER_LAUNCH_DELAY_S:-180}}"
export DEXPBT_STARTUP_TIMEOUT_S="${{DEXPBT_STARTUP_TIMEOUT_S:-1200}}"
export DEXPBT_STARTUP_MAX_RESTARTS="${{DEXPBT_STARTUP_MAX_RESTARTS:-3}}"
export NUM_POLICIES=6
export DEXPBT_GPUS="${{DEXPBT_GPUS:-0,1,2,3,4,5}}"

mkdir -p "$RUN_DIR"
cp "$(dirname "$0")/{arm_lower}.config.json" "$RUN_DIR/ARM_CONFIG.json"
echo "$RUN_DIR" > "$RUN_DIR/RUN_DIR.txt"

source "${{RUNTIME_ROOT}}/scripts/env_dexpbt.sh"
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1

cd "${{RUNTIME_ROOT}}/src/dexpbt_isaaclab"

# STATE-FAITHFUL BOOTSTRAP (fail closed): restore the source run's full
# stall state BEFORE anything launches — curriculum_control gate state
# (msa_* cleared, everything else intact), rebuilt observer memory
# (observer_state_seed/), per-policy curriculum_resume_snapshot copies,
# STATE_FIDELITY.json + STATE_FINGERPRINT.json. An empty curriculum_control
# at worker start falls back to the ungated LEGACY curriculum (the exact
# 2026-07-19 smoke failure) — never launch without this step.
python -m dexpbt_isaaclab.msa.state_bootstrap \\
  --source-run "$SOURCE_RUN" \\
  --run-dir "$RUN_DIR" \\
  --workspace "$RUN_DIR/dexpbt_8gpu_ws" \\
  | tee -a "$LOG"
test "${{PIPESTATUS[0]}}" -eq 0 || {{
  echo "REFUSING TO LAUNCH: state bootstrap failed (fail closed)" | tee -a "$LOG"
  exit 3
}}

# Fingerprint validation: the restored state on disk hashes to the recorded
# fingerprint and no consumed source file changed since the bootstrap.
python -m dexpbt_isaaclab.msa.state_bootstrap \\
  --verify \\
  --run-dir "$RUN_DIR" \\
  --workspace "$RUN_DIR/dexpbt_8gpu_ws" \\
  | tee -a "$LOG"
test "${{PIPESTATUS[0]}}" -eq 0 || {{
  echo "REFUSING TO LAUNCH: state fingerprint validation failed" | tee -a "$LOG"
  exit 3
}}

# Fail-closed bootstrap verification: checkpoint existence + hash + worker
# mapping + manifest fields. Any failure aborts BEFORE anything launches.
python -m dexpbt_isaaclab.msa.continuation_guard \\
  --config "$RUN_DIR/ARM_CONFIG.json" --verify-only \\
  | tee -a "$LOG"
test "${{PIPESTATUS[0]}}" -eq 0 || {{
  echo "REFUSING TO LAUNCH: bootstrap verification failed" | tee -a "$LOG"
  exit 3
}}

nohup bash "${{RUNTIME_ROOT}}/scripts/launch_8gpu_dexpbt.sh" \\
  "$RUN_DIR" \\
  --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \\
  --num-envs 8192 \\
  --max-iterations {max_iter} \\
  --pbt-interval-steps 20000000 \\
  --pbt-start-after 200000000 \\
  --pbt-initial-delay 50000000 \\
  --profile-per-policy "{profile_list}" \\
  --recover-checkpoint-per-policy "$RECOVER_LIST" \\
  --tolerance-resume-per-policy "$TOLERANCE_LIST" \\
  --curriculum-clock-resume-per-policy "$CLOCK_LIST" \\
  --extra \\
  --seed "$BASE_SEED" \\
{base_flags}  >>"$LOG" 2>&1 &

echo $! > "$RUN_DIR/nohup.pid"

# Continuation guard: (a) six-worker READINESS BARRIER + STALL REPRODUCTION
# GATE — each worker's first committed tick must confirm the structured
# RESUME_APPLIED.json tolerance resume, stay within ONE mastery shrink of
# its source tolerance (two shrinks in one 80M interval = ungated legacy
# fallback -> GUARD_FAILED), and keep the restored control state alive
# (no default clobber -> STATE_FIDELITY_FAILED); >= 4/6 workers must hold
# EXACTLY their source tolerance or STATE_FIDELITY_FAILED.json stops the
# arm; only then FIRST_TICK_VALIDATED.json is written;
# (b) matched worker-local cap — target = baseline + {cap_frames} (6
# committed ticks at the REAL 80M cadence); park at first committed frame
# >= target; overshoot beyond one tick (80M) of slack fails the arm; all
# six parked -> ARM_COMPLETE.json (normal completion) and full stop with
# SIGTERM -> 20s grace -> SIGKILL escalation;
# (c) while MSA_BUSY.json exists (in-flight probe cycle) park/cap actions
# are deferred (cap_pending) and re-evaluated on fresh committed metas
# once the sidecar clears the marker.
nohup python -u -m dexpbt_isaaclab.msa.continuation_guard \\
  --config "$RUN_DIR/ARM_CONFIG.json" \\
  --run-dir "$RUN_DIR" \\
  --workspace "$RUN_DIR/dexpbt_8gpu_ws" \\
  --cap-frames {cap_frames} \\
  >>"$RUN_DIR/continuation_guard.log" 2>&1 &
echo $! > "$RUN_DIR/continuation_guard.pid"
{sidecar}echo "launched {arm}: run_dir=$RUN_DIR"
"""

_SIDECAR_BLOCK = """
# MSA probe sidecar: drives the probe state machine over the shared
# curriculum_control chain (committed-tick views, phase-barrier holds with
# stopped-state verification + MSA_HOLDS.json persistence, heterogeneous
# per-worker tau_plus pins via msa_forced_success_tolerance, policy-only
# evals in subprocesses with bounded per-device concurrency).
# - the sidecar WAITS for BOTH the guard's FIRST_TICK_VALIDATED.json (6/6
#   readiness barrier) AND HOLD_SMOKE_PASSED.json (written only by a fully
#   successful run_hold_smoke.sh) before starting or resuming any cycle —
#   it may be launched early but idles at the gate without touching
#   MSA_HOLDS.json (execution order: launch arm -> 6/6 readiness ->
#   run_hold_smoke.sh -> PASSED marker -> first N=4 cycle);
# - MSA_EVAL_DEVICES defaults to cuda:6,cuda:7: trainers occupy DEXPBT_GPUS
#   (0-5); eval devices are REQUIRED and never default onto cuda:0;
# - MSA_NUM_SEEDS defaults to 4 = EXECUTION SMOKE (validates the machinery
#   only, NEVER a method conclusion). Pilot N=16 must be set explicitly;
#   formal N=256 is formal-only (~345h on 2 GPUs — do not).
nohup python -u -m dexpbt_isaaclab.msa.sidecar \\
  --workspace "$RUN_DIR/dexpbt_8gpu_ws" \\
  --arm {arm} \\
  --rho-threshold {rho} \\
  --num-seeds "${{MSA_NUM_SEEDS:-4}}" \\
  --eval-devices "${{MSA_EVAL_DEVICES:-cuda:6,cuda:7}}" \\
  --readiness-marker "$RUN_DIR/FIRST_TICK_VALIDATED.json" \\
  --hold-smoke-marker "$RUN_DIR/HOLD_SMOKE_PASSED.json" \\
  --commit-max-wait-s "${{MSA_COMMIT_MAX_WAIT_S:-3600}}" \\
  {permuted_flag}>>"$RUN_DIR/msa_sidecar.log" 2>&1 &
echo $! > "$RUN_DIR/msa_sidecar.pid"

# External hold watchdog (independent of the sidecar's finally block): if
# the sidecar dies while workers are SIGSTOPped (MSA_HOLDS.json non-empty),
# it SIGCONTs all recorded PIDs, writes MSA_SIDECAR_DIED.json and stops the
# whole arm — never uncontrolled continuation training.
nohup python -u -m dexpbt_isaaclab.msa.hold_watchdog \\
  --run-dir "$RUN_DIR" \\
  >>"$RUN_DIR/hold_watchdog.log" 2>&1 &
echo $! > "$RUN_DIR/hold_watchdog.pid"
"""

_HOLD_SMOKE_TEMPLATE = """#!/usr/bin/env bash
# Single-worker HOLD SMOKE for a running MSA arm. Prepared but NOT executed.
# Validates SIGSTOP/SIGCONT against a live Isaac trainer BEFORE any probe
# cycle: waits for the guard's readiness marker, holds ONE worker, verifies
# its committed frame is frozen for the hold window, releases, verifies the
# next committed tick appears. No tolerance writes, no MSA cycle, no eval.
# Result: <run_dir>/HOLD_SMOKE_RESULT.json (release guaranteed either way).
# GATE: HOLD_SMOKE_PASSED.json is written atomically ONLY when the frame
# stayed frozen, the next committed tick appeared after release AND no
# residual holds remain — the formal sidecar refuses to start any cycle
# until it exists. Any failure writes HOLD_SMOKE_FAILED.json and
# fail-closed stops the whole arm.
set -euo pipefail

RUN_DIR="${1:?usage: run_hold_smoke.sh <run_dir> [worker] [seconds]}"
WORKER="${2:-0}"
HOLD_SECONDS="${3:-60}"
RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"

source "${RUNTIME_ROOT}/scripts/env_dexpbt.sh"
cd "${RUNTIME_ROOT}/src/dexpbt_isaaclab"

python -u -m dexpbt_isaaclab.msa.sidecar \\
  --workspace "$RUN_DIR/dexpbt_8gpu_ws" \\
  --arm MSA_STATEFUL \\
  --readiness-marker "$RUN_DIR/FIRST_TICK_VALIDATED.json" \\
  --hold-smoke-only \\
  --hold-smoke-worker "$WORKER" \\
  --hold-smoke-seconds "$HOLD_SECONDS"
echo "hold smoke result: $RUN_DIR/HOLD_SMOKE_RESULT.json"
"""


def _base_flags_block(arm: str, original_launcher: Path) -> str:
    """Extract the --extra flag block verbatim from the original launcher and
    apply the arm's diff. Keeps every original numeric gate identical."""
    text = original_launcher.read_text()
    start = text.index("--pbt_ranking_metric")
    end = text.index('"${SHARED_WM_HEAD_FLAGS[@]}"')
    block = text[text.rindex("\n", 0, start) + 1:end]
    shared_flags = (
        "  --seq_wm_head_balance_enable \\\n"
        "  --seq_wm_head_pos_weight_auto \\\n"
        "  --seq_wm_head_pos_weight_max 10.0 \\\n"
        "  --seq_wm_lambda_horizon_success 0.5 \\\n"
        "  --seq_wm_lambda_drop_risk 0.5 \\\n"
        "  --seq_wm_lambda_recovery 0.5 \\\n"
        "  --seq_wm_replay_prioritize_floor_recovery \\\n"
        "  --seq_wm_floor_aware_weighting \\\n"
        "  --eir_candidate_gate_mode profile_soft \\\n"
        "  --eir_candidate_min_weight_for_retrieval 0.05 \\\n")
    block = block + shared_flags
    if arm == "GATE_RELAX_STATEFUL":
        old = f"{GATE_RELAX_FLAG} {GATE_RELAX_OLD}"
        new = f"{GATE_RELAX_FLAG} {GATE_RELAX_NEW}"
        if old not in block:
            raise ValueError(f"expected '{old}' in base flag block")
        block = block.replace(old, new)
    return block


def _validate_gate_relax_profile() -> None:
    """Fail closed if the dedicated GATE_RELAX profile is missing or wrong
    (profile expansion would silently revert a CLI-only override)."""
    from dexpbt_isaaclab.world_model.wm_strategy_profiles import (
        PROFILE_TABLES,
    )
    table = PROFILE_TABLES.get(GATE_RELAX_PROFILE)
    if table is None:
        raise ValueError(f"profile {GATE_RELAX_PROFILE} not registered")
    got = table.get("late_floor_retention_min_retention")
    if abs(float(got) - float(GATE_RELAX_NEW)) > 1e-12:
        raise ValueError(
            f"{GATE_RELAX_PROFILE} table has "
            f"late_floor_retention_min_retention={got}, expected "
            f"{GATE_RELAX_NEW}")
    base = PROFILE_TABLES[BASE_PROFILE]
    extra = {k for k in table if table[k] != base.get(k)}
    if extra != {"late_floor_retention_min_retention"}:
        raise ValueError(f"{GATE_RELAX_PROFILE} diff vs {BASE_PROFILE} must "
                         f"be exactly one key, got {sorted(extra)}")


def _clock_list_from_source(source_run: Path, num_workers: int = 6) -> str:
    """Per-policy 'lcu:fsr' env-clock resume pairs from the source run's
    curriculum_resume_snapshot.json files (fail closed on any missing)."""
    pairs = []
    for w in range(num_workers):
        snap_path = (source_run / f"policy_{w:02d}"
                     / "curriculum_resume_snapshot.json")
        if not snap_path.is_file():
            raise FileNotFoundError(
                f"source resume snapshot missing: {snap_path}")
        clock = env_clock_from_snapshot(json.loads(snap_path.read_text()))
        pairs.append(f"{clock['last_curriculum_update']}:"
                     f"{clock['frame_since_restart']}")
    return ",".join(pairs)


def write_arm_launchers(out_dir: Path, manifest_csv: Path,
                        original_launcher: Path,
                        source_run: Path) -> list[Path]:
    """Write <arm>.sh + <arm>.config.json for all four arms. Never executes."""
    _validate_gate_relax_profile()
    tips = resolve_tip_checkpoints(manifest_csv)
    recover_list = ",".join(t.checkpoint_path for t in tips)
    tolerance_list = ",".join(f"{t.live_tolerance!r}" for t in tips)
    clock_list = _clock_list_from_source(source_run)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for arm in ARM_NAMES:
        cfg = arm_config(arm, tips, str(source_run))
        cfg_path = out_dir / f"{arm.lower()}.config.json"
        cfg_path.write_text(json.dumps(cfg, indent=2))
        written.append(cfg_path)
        sidecar = ""
        if arm in MSA_ARM_NAMES:
            sidecar = _SIDECAR_BLOCK.format(
                arm=arm, rho=MSA_RHO_THRESHOLD,
                permuted_flag=("--permuted-route "
                               if arm == "MSA_PERMUTED_STATEFUL" else ""))
        wm_profile = (GATE_RELAX_PROFILE if arm == "GATE_RELAX_STATEFUL"
                      else BASE_PROFILE)
        script = _LAUNCHER_TEMPLATE.format(
            arm=arm, arm_lower=arm.lower(), recover_list=recover_list,
            tolerance_list=tolerance_list,
            clock_list=clock_list,
            source_run=str(source_run),
            cap_frames=CONTINUATION_FRAMES,
            max_iter=CONTINUATION_MAX_ITER,
            profile_list=",".join([wm_profile] * 6),
            base_flags=_base_flags_block(arm, original_launcher),
            sidecar=sidecar)
        sh_path = out_dir / f"start_{arm.lower()}.sh"
        sh_path.write_text(script)
        sh_path.chmod(0o755)
        written.append(sh_path)
    # single-worker hold smoke launcher (prepared, never executed here)
    smoke_path = out_dir / "run_hold_smoke.sh"
    smoke_path.write_text(_HOLD_SMOKE_TEMPLATE)
    smoke_path.chmod(0o755)
    written.append(smoke_path)
    # true per-cycle eval workload (27 cells) at each budget on 2 eval GPUs
    cost_path = out_dir / "EVAL_COST_ESTIMATE.json"
    cost_path.write_text(json.dumps({
        "note": "one full MSA probe cycle; one Isaac process per "
                "(cell, seed); at most one process per eval device",
        "estimates": [
            eval_cost_estimate(SMOKE_NUM_SEEDS, 2),
            eval_cost_estimate(PILOT_NUM_SEEDS, 2),
            eval_cost_estimate(FORMAL_NUM_SEEDS, 2),
        ],
        "verdicts": {
            "N=4  x 2 GPUs": "~5.4 h  — execution smoke (machinery only)",
            "N=16 x 2 GPUs": "~21.6 h — pilot (explicit opt-in)",
            "N=256 x 2 GPUs": "~345.6 h — NOT acceptable without "
                              "per-process batching / seed vectorization",
        },
    }, indent=2))
    written.append(cost_path)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--original-launcher", required=True)
    ap.add_argument("--source-run", required=True,
                    help="source run whose full stall state is restored by "
                         "the state-faithful bootstrap at launch")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    written = write_arm_launchers(Path(args.out_dir), Path(args.manifest_csv),
                                  Path(args.original_launcher),
                                  Path(args.source_run))
    for p in written:
        print(p)
    print("NOTE: launchers were generated but NOT executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
