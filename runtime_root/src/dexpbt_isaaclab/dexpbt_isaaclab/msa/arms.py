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
                                "cuda:4,cuda:5}: on this 6-GPU machine the "
                                "eval devices are SHARED with trainers 4/5 "
                                "(those trainers are SIGSTOPped during "
                                "every eval barrier but keep their VRAM "
                                "allocated); never the trainers' cuda:0; "
                                "at most one Isaac eval process per "
                                "device; actual devices recorded in "
                                "EVAL_DEVICES.json",
            "hold_smoke_note": "the formal sidecar requires the DISPOSABLE "
                               "start_stateful_hold_smoke.sh certificate "
                               "(HOLD_SMOKE_CERTIFICATE.json: git commit, "
                               "task, profile, GPU model, driver must "
                               "match); the in-arm run_hold_smoke.sh is "
                               "execution-smoke only (it advances the "
                               "released worker one extra committed tick "
                               "past the matched boundary)",
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
LINEAGE_LIST="{lineage_list}"

export DEXPBT_MAX_ACTIVE="${{DEXPBT_MAX_ACTIVE:-6}}"
export DEXPBT_INTER_LAUNCH_DELAY_S="${{DEXPBT_INTER_LAUNCH_DELAY_S:-180}}"
export DEXPBT_STARTUP_TIMEOUT_S="${{DEXPBT_STARTUP_TIMEOUT_S:-1200}}"
export DEXPBT_STARTUP_MAX_RESTARTS="${{DEXPBT_STARTUP_MAX_RESTARTS:-3}}"
# launch_8gpu_dexpbt.sh reads DEXPBT_NUM_POLICIES (NOT NUM_POLICIES)
export DEXPBT_NUM_POLICIES=6
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
  --lineage-resume-per-policy "$LINEAGE_LIST" \\
  --extra \\
  --seed "$BASE_SEED" \\
{base_flags}  >>"$LOG" 2>&1 &

echo $! > "$RUN_DIR/nohup.pid"

# Continuation guard: (a) six-worker TRUE READINESS BARRIER + STALL
# REPRODUCTION GATE — each worker's first committed tick is validated
# (structured RESUME_APPLIED.json tolerance resume; at most ONE mastery
# shrink of its source tolerance — two shrinks in one 80M interval =
# ungated legacy fallback -> GUARD_FAILED; restored control state alive,
# no default clobber -> STATE_FIDELITY_FAILED) and the worker is then
# IMMEDIATELY SIGSTOP-frozen at that boundary (READINESS_HOLDS.json);
# validated workers stay frozen until 6/6 are ready; >= 4/6 workers must
# hold EXACTLY their source tolerance or STATE_FIDELITY_FAILED.json stops
# the arm; only then FIRST_TICK_VALIDATED.json is written. Non-MSA arms
# release all six SIMULTANEOUSLY; MSA arms leave the holds for the
# sidecar to adopt (READINESS_HOLDS.json -> MSA_HOLDS.json), so the first
# PRE snapshot comes from exactly the validated frame;
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

# External hold watchdog (ALL stateful arms), independent of the guard's
# and the sidecar's finally blocks. It covers the WHOLE hold lifecycle:
# - readiness phase: if the guard dies while workers are SIGSTOP-frozen in
#   READINESS_HOLDS.json (before the marker, or — for MSA arms — after the
#   marker but before the sidecar adopts), it SIGCONTs the frozen workers,
#   writes GUARD_DIED_DURING_READINESS.json and stops the whole arm;
# - cycle phase (MSA): if the sidecar dies with MSA_HOLDS.json non-empty OR
#   MSA_BUSY.json present (even mid-COMMITTING with no holds), it releases
#   the held workers, writes MSA_SIDECAR_DIED.json and stops the arm.
# It never idle-exits while readiness holds exist and fails closed on an
# unreadable holds file (no uncontrolled continuation training).
nohup python -u -m dexpbt_isaaclab.msa.hold_watchdog \\
  --run-dir "$RUN_DIR" \\
  >>"$RUN_DIR/hold_watchdog.log" 2>&1 &
echo $! > "$RUN_DIR/hold_watchdog.pid"
{sidecar}echo "launched {arm}: run_dir=$RUN_DIR"
"""

_SIDECAR_BLOCK = """
# MSA probe sidecar: drives the probe state machine over the shared
# curriculum_control chain (committed-tick views, phase-barrier holds with
# stopped-state verification + MSA_HOLDS.json persistence, heterogeneous
# per-worker tau_plus pins via msa_forced_success_tolerance, policy-only
# evals in subprocesses with bounded per-device concurrency).
# - the sidecar WAITS for the guard's FIRST_TICK_VALIDATED.json (6/6 TRUE
#   readiness barrier: every worker is SIGSTOP-frozen at its validated
#   first tick) and then ADOPTS the guard's READINESS_HOLDS.json (verifies
#   stopped state + committed frame == validated frame, converts it to
#   MSA_HOLDS.json) so the first PRE snapshot comes from EXACTLY the
#   matched intervention boundary — workers are never released in between;
# - the hold-smoke gate is the DISPOSABLE certificate: run
#   start_stateful_hold_smoke.sh (same code / task / 8192 envs / shared
#   eval GPU) BEFORE launching this arm; the sidecar validates
#   HOLD_SMOKE_CERTIFICATE.json (git commit, task, profile, GPU model,
#   driver) at startup and fail-closed stops the arm on any mismatch.
#   NEVER run the in-arm run_hold_smoke.sh in a formal arm: it releases
#   the target worker for an extra committed tick and breaks the matched
#   boundary (execution smoke only);
# - SIX-GPU MACHINE: trainers occupy DEXPBT_GPUS (cuda:0-5); eval shares
#   cuda:4,cuda:5 with the trainers — safe because the GPU4/5 trainers are
#   SIGSTOPped during every eval barrier (they keep their VRAM allocated;
#   budget headroom on those two GPUs is required); the ACTUAL eval
#   devices are recorded in $RUN_DIR/EVAL_DEVICES.json;
# - MSA_NUM_SEEDS defaults to 4 = EXECUTION SMOKE (validates the machinery
#   only, NEVER a method conclusion). Pilot N=16 must be set explicitly;
#   formal N=256 is formal-only (~345h on 2 GPUs — do not).
MSA_EVAL_DEVICES_ACTUAL="${{MSA_EVAL_DEVICES:-cuda:4,cuda:5}}"
MSA_HOLD_SMOKE_CERT_ACTUAL="${{MSA_HOLD_SMOKE_CERT:-{cert_default}}}"
if [[ ! -f "$MSA_HOLD_SMOKE_CERT_ACTUAL" ]]; then
  echo "REFUSING TO LAUNCH SIDECAR: hold smoke certificate missing:" | tee -a "$LOG"
  echo "  $MSA_HOLD_SMOKE_CERT_ACTUAL" | tee -a "$LOG"
  echo "  run start_stateful_hold_smoke.sh first (disposable GPU smoke)" | tee -a "$LOG"
  exit 3
fi
cat > "$RUN_DIR/EVAL_DEVICES.json" <<EOF
{{
  "eval_devices": "$MSA_EVAL_DEVICES_ACTUAL",
  "trainer_gpus": "$DEXPBT_GPUS",
  "shared_gpu_note": "trainers on the eval GPUs are SIGSTOPped during eval barriers but keep their VRAM allocated",
  "hold_smoke_certificate": "$MSA_HOLD_SMOKE_CERT_ACTUAL"
}}
EOF
nohup python -u -m dexpbt_isaaclab.msa.sidecar \\
  --workspace "$RUN_DIR/dexpbt_8gpu_ws" \\
  --arm {arm} \\
  --rho-threshold {rho} \\
  --num-seeds "${{MSA_NUM_SEEDS:-4}}" \\
  --eval-devices "$MSA_EVAL_DEVICES_ACTUAL" \\
  --readiness-marker "$RUN_DIR/FIRST_TICK_VALIDATED.json" \\
  --hold-smoke-certificate "$MSA_HOLD_SMOKE_CERT_ACTUAL" \\
  --wm-profile "{profile}" \\
  --commit-max-wait-s "${{MSA_COMMIT_MAX_WAIT_S:-3600}}" \\
  {permuted_flag}>>"$RUN_DIR/msa_sidecar.log" 2>&1 &
echo $! > "$RUN_DIR/msa_sidecar.pid"
# NOTE: the external hold watchdog is started once for ALL stateful arms
# in the main launcher body (above), so it already covers this sidecar's
# MSA_HOLDS.json / MSA_BUSY.json in addition to the readiness barrier.
"""

_HOLD_SMOKE_TEMPLATE = """#!/usr/bin/env bash
# Single-worker HOLD SMOKE for a running arm — EXECUTION SMOKE ONLY.
#
# FORBIDDEN in the formal four-arm comparison: this smoke RELEASES the
# target worker and waits for one extra committed tick, which advances the
# worker past its matched intervention boundary. The formal MSA arms use
# the DISPOSABLE start_stateful_hold_smoke.sh certificate instead
# (HOLD_SMOKE_CERTIFICATE.json), validated by the sidecar at startup.
#
# Default worker is 4: it trains on a SHARED eval GPU (cuda:4), so the
# execution smoke exercises the shared-GPU hold path.
#
# Validates SIGSTOP/SIGCONT against a live Isaac trainer: waits for the
# guard's readiness marker, holds ONE worker, verifies its committed frame
# is frozen for the hold window, releases, verifies the next committed
# tick appears. No tolerance writes, no MSA cycle, no eval.
# Result: <run_dir>/HOLD_SMOKE_RESULT.json (release guaranteed either way).
# HOLD_SMOKE_PASSED.json is written atomically ONLY when the frame stayed
# frozen, the next committed tick appeared after release AND no residual
# holds remain. Any failure writes HOLD_SMOKE_FAILED.json and fail-closed
# stops the whole arm.
set -euo pipefail

RUN_DIR="${1:?usage: run_hold_smoke.sh <run_dir> [worker] [seconds]}"
WORKER="${2:-4}"
HOLD_SECONDS="${3:-60}"
RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"

echo "WARNING: execution smoke only — the released worker will advance one"
echo "extra committed tick; NEVER use inside the formal four-arm comparison."

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

_DISPOSABLE_SMOKE_TEMPLATE = """#!/usr/bin/env bash
# DISPOSABLE GPU hold smoke — produces HOLD_SMOKE_CERTIFICATE.json.
# GENERATED by dexpbt_isaaclab.msa.arms — prepared but NOT executed.
#
# Runs OUTSIDE the formal arms: launches ONE throwaway worker with the
# SAME code, Isaac task, wm profile and 8192 envs on the shared eval GPU
# (default cuda:4), waits for its first committed tick, SIGSTOPs it,
# verifies the committed frame is frozen, then — WHILE the trainer is
# frozen and still holding its VRAM — launches one real policy-only eval
# cell (N=1) on the SAME GPU and requires it to exit cleanly (no CUDA
# OOM / Isaac startup failure), SIGCONTs, verifies the next committed
# tick appears — then writes the certificate (eval_coexistence_ok=true)
# and tears the whole disposable run down. It produces NO method results;
# the formal MSA launchers validate the certificate (eval_coexistence_ok,
# git commit, task, profile, GPU model, driver) at sidecar startup and
# refuse to run without it.
set -euo pipefail

if [[ "${{1:-}}" != "--i-know-what-i-am-doing" ]]; then
  echo "REFUSING TO LAUNCH: pass --i-know-what-i-am-doing"
  echo "(this starts ONE disposable GPU training worker on GPU ${{SMOKE_GPU:-4}})"
  exit 2
fi

RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${{RESULTS_ROOT}}/runs/msa_hold_smoke_disposable_${{DATE}}"
LOG="${{RUN_DIR}}/launch_master.log"
CERT_OUT="${{MSA_HOLD_SMOKE_CERT:-{cert_default}}}"
SMOKE_GPU="${{SMOKE_GPU:-4}}"
HOLD_SECONDS="${{HOLD_SECONDS:-60}}"

export DEXPBT_MAX_ACTIVE=1
# launch_8gpu_dexpbt.sh reads DEXPBT_NUM_POLICIES (NOT NUM_POLICIES)
export DEXPBT_NUM_POLICIES=1
export DEXPBT_GPUS="$SMOKE_GPU"
export DEXPBT_STARTUP_TIMEOUT_S="${{DEXPBT_STARTUP_TIMEOUT_S:-1200}}"

mkdir -p "$RUN_DIR"
source "${{RUNTIME_ROOT}}/scripts/env_dexpbt.sh"
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
cd "${{RUNTIME_ROOT}}/src/dexpbt_isaaclab"

# one throwaway worker: same task / profile / env count as the formal
# arms; recovered from worker-4 tip so the trainer shape is realistic
nohup bash "${{RUNTIME_ROOT}}/scripts/launch_8gpu_dexpbt.sh" \\
  "$RUN_DIR" \\
  --task {task} \\
  --num-envs 8192 \\
  --max-iterations {max_iter} \\
  --pbt-interval-steps 20000000 \\
  --pbt-start-after 200000000 \\
  --pbt-initial-delay 50000000 \\
  --profile-per-policy "{profile}" \\
  --recover-checkpoint-per-policy "{smoke_ckpt}" \\
  --tolerance-resume-per-policy "{smoke_tol}" \\
  --lineage-resume-per-policy "{smoke_lineage}" \\
  --extra \\
  --seed 42 \\
{base_flags}  >>"$LOG" 2>&1 &
echo $! > "$RUN_DIR/nohup.pid"

# certificate smoke: wait first committed tick -> hold -> verify frozen ->
# release -> verify next tick -> write HOLD_SMOKE_CERTIFICATE.json
set +e
python -u -m dexpbt_isaaclab.msa.sidecar \\
  --workspace "$RUN_DIR/dexpbt_8gpu_ws" \\
  --arm MSA_STATEFUL \\
  --task {task} \\
  --readiness-marker "$RUN_DIR/UNUSED.json" \\
  --certificate-smoke \\
  --hold-smoke-worker 0 \\
  --hold-smoke-seconds "$HOLD_SECONDS" \\
  --cert-out "$CERT_OUT" \\
  --cert-gpu-index "$SMOKE_GPU" \\
  --cert-eval-num-envs "${{CERT_EVAL_NUM_ENVS:-512}}" \\
  --cert-eval-timeout-s "${{CERT_EVAL_TIMEOUT_S:-1800}}" \\
  --wm-profile "{profile}" \\
  2>&1 | tee -a "$LOG"
RC="${{PIPESTATUS[0]}}"
set -e

# tear the disposable run down COMPLETELY (children-first escalation);
# it must never keep training or occupy the GPU
python - "$RUN_DIR" <<'PY'
import sys
from pathlib import Path
from dexpbt_isaaclab.msa.continuation_guard import stop_arm
stop_arm(Path(sys.argv[1]), "disposable hold smoke torn down",
         marker_name=None, num_workers=1)
PY

echo "certificate: $CERT_OUT (ok must be true)"
exit "$RC"
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


def _lineage_list_from_tips(tips: list["TipCheckpoint"]) -> str:
    """Per-policy 'lineage_id:lineage_start_pf:weight_age_frame' resume
    triples from each tip checkpoint's sibling meta JSON (fail closed on a
    missing meta, a missing field, or a meta/manifest lineage mismatch).

    Without this a warm-started worker mints a FRESH lineage uuid at its
    first committed tick (fresh workspace = no prior meta for the
    observer's fallback) and the continuation guard rejects it against the
    manifest lineage.
    """
    triples = []
    for t in tips:
        meta_path = Path(t.checkpoint_path).with_suffix(".json")
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"worker {t.worker_id}: tip meta missing: {meta_path}")
        meta = json.loads(meta_path.read_text())
        lineage = str(meta.get("lineage_id") or "")
        if not lineage:
            raise ValueError(
                f"worker {t.worker_id}: tip meta has no lineage_id")
        if t.lineage_id and lineage != t.lineage_id:
            raise ValueError(
                f"worker {t.worker_id}: tip meta lineage {lineage} != "
                f"manifest lineage {t.lineage_id}")
        start_pf = meta.get("lineage_start_pf")
        weight_age = meta.get("weight_age_frame")
        if start_pf is None or weight_age is None:
            raise ValueError(
                f"worker {t.worker_id}: tip meta missing lineage_start_pf "
                f"or weight_age_frame")
        triples.append(f"{lineage}:{int(start_pf)}:{int(weight_age)}")
    return ",".join(triples)


def write_arm_launchers(out_dir: Path, manifest_csv: Path,
                        original_launcher: Path,
                        source_run: Path) -> list[Path]:
    """Write <arm>.sh + <arm>.config.json for all four arms. Never executes."""
    _validate_gate_relax_profile()
    tips = resolve_tip_checkpoints(manifest_csv)
    recover_list = ",".join(t.checkpoint_path for t in tips)
    tolerance_list = ",".join(f"{t.live_tolerance!r}" for t in tips)
    clock_list = _clock_list_from_source(source_run)
    lineage_list = _lineage_list_from_tips(tips)
    out_dir.mkdir(parents=True, exist_ok=True)
    cert_default = str((out_dir / "HOLD_SMOKE_CERTIFICATE.json").resolve())
    written: list[Path] = []
    for arm in ARM_NAMES:
        cfg = arm_config(arm, tips, str(source_run))
        cfg_path = out_dir / f"{arm.lower()}.config.json"
        cfg_path.write_text(json.dumps(cfg, indent=2))
        written.append(cfg_path)
        wm_profile = (GATE_RELAX_PROFILE if arm == "GATE_RELAX_STATEFUL"
                      else BASE_PROFILE)
        sidecar = ""
        if arm in MSA_ARM_NAMES:
            sidecar = _SIDECAR_BLOCK.format(
                arm=arm, rho=MSA_RHO_THRESHOLD,
                profile=wm_profile,
                cert_default=cert_default,
                permuted_flag=("--permuted-route "
                               if arm == "MSA_PERMUTED_STATEFUL" else ""))
        script = _LAUNCHER_TEMPLATE.format(
            arm=arm, arm_lower=arm.lower(), recover_list=recover_list,
            tolerance_list=tolerance_list,
            clock_list=clock_list,
            lineage_list=lineage_list,
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
    # single-worker hold smoke launcher — EXECUTION SMOKE ONLY (never part
    # of the formal four-arm comparison)
    smoke_path = out_dir / "run_hold_smoke.sh"
    smoke_path.write_text(_HOLD_SMOKE_TEMPLATE)
    smoke_path.chmod(0o755)
    written.append(smoke_path)
    # DISPOSABLE certificate smoke launcher (one throwaway worker on the
    # shared eval GPU; produces HOLD_SMOKE_CERTIFICATE.json)
    smoke_tip = next(t for t in tips if t.worker_id == 4)
    disposable_path = out_dir / "start_stateful_hold_smoke.sh"
    disposable_path.write_text(_DISPOSABLE_SMOKE_TEMPLATE.format(
        cert_default=cert_default,
        task="DexPBT-Regrasping-KukaAllegro-Direct-v0",
        max_iter=CONTINUATION_MAX_ITER,
        profile=BASE_PROFILE,
        smoke_ckpt=smoke_tip.checkpoint_path,
        smoke_tol=f"{smoke_tip.live_tolerance!r}",
        smoke_lineage=_lineage_list_from_tips([smoke_tip]),
        base_flags=_base_flags_block("BASE_STATEFUL", original_launcher)))
    disposable_path.chmod(0o755)
    written.append(disposable_path)
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
