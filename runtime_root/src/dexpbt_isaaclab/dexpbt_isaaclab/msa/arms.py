"""Continuation arm configs + launcher generation for base_retention_msa_probe.

Four arms, all resuming from the SAME six 5B final checkpoints (resolved
from the CCRS checkpoint manifest, never guessed), matched seeds, 100M
worker-local continuation frames each:

- BASE_CONTINUATION   : the original base-retention flags, unchanged.
- GATE_RELAX          : ONLY late_floor_retention_min_retention 0.88 -> 0.75
                        (the pre-existing hard minimum); everything else
                        identical.
- MSA                 : original numeric gates unchanged; the MSA probe
                        sidecar replaces the cross-tolerance retention read
                        with same-tau rho.
- MSA_PERMUTED_ROUTE  : MSA with the pre-registered permuted second-half
                        routing (negative control for exposure-only gains).

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
from dexpbt_isaaclab.msa.eval_bridge import FORMAL_NUM_SEEDS, build_seed_manifest
from dexpbt_isaaclab.msa.provenance import EXPOSURE_WINDOW_FRAMES

ARM_NAMES = ("BASE_CONTINUATION", "GATE_RELAX", "MSA", "MSA_PERMUTED_ROUTE")

# Continuation budget: 100M worker-local frames on top of the 5B tips.
CONTINUATION_FRAMES = 100_000_000
# The strict-5B run reached 5B population frames in 44500 RL epochs
# (~112,360 frames/epoch at 8192 envs); +100M worker-local frames ~ +890.
BASE_MAX_ITER = 44_500
CONTINUATION_MAX_ITER = BASE_MAX_ITER + 890

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


def resolve_tip_checkpoints(manifest_csv: Path,
                            num_policies: int = 6) -> list[TipCheckpoint]:
    """Resolve the six 5B tip checkpoints from the CCRS manifest.

    Fails closed: every worker 0..num_policies-1 must have exactly one row
    whose roles include 'tip' and whose file exists on disk.
    """
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
            lineage_id=r.get("lineage_id") or ""))
    return tips


def arm_config(arm: str, tips: list[TipCheckpoint], base_seed: int = 42) -> dict:
    """Machine-readable config for one arm (written next to its launcher)."""
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown arm {arm}")
    manifest_id, seeds = build_seed_manifest(FORMAL_NUM_SEEDS)
    wm_profile = GATE_RELAX_PROFILE if arm == "GATE_RELAX" else BASE_PROFILE
    cfg: dict = {
        "arm": arm,
        "profile": "base_retention_msa_probe",
        "wm_strategy_profile": wm_profile,
        "resume_checkpoints": {
            f"p{t.worker_id}": {
                "path": t.checkpoint_path,
                "iteration": t.iteration,
                "frame": t.frame,
                "live_tolerance": t.live_tolerance,
                "lineage_id": t.lineage_id,
            } for t in tips
        },
        "base_seed": base_seed,           # matched across all arms
        "continuation_frames_per_worker": CONTINUATION_FRAMES,
        "max_iterations": CONTINUATION_MAX_ITER,
        "frame_ledger": {
            "monotonic": True,
            "probe_frames_charged": True,
            "note": "probe packets train through the normal supervisor loop "
                    "under forced tolerance; they consume the same 100M "
                    "budget as ordinary continuation frames",
        },
        "original_launcher": ORIGINAL_LAUNCHER,
        "flag_diff_vs_base": [],
    }
    if arm == "GATE_RELAX":
        cfg["flag_diff_vs_base"] = [
            {"flag": GATE_RELAX_FLAG, "old": GATE_RELAX_OLD,
             "new": GATE_RELAX_NEW,
             "note": "already the configured hard minimum "
                     "(late_floor_retention_hard_min_retention); applied via "
                     f"the dedicated {GATE_RELAX_PROFILE} profile table "
                     "because profile expansion is authoritative over CLI"}]
    if arm in ("MSA", "MSA_PERMUTED_ROUTE"):
        cfg["msa"] = {
            "tau_plus_rule": "max(0.010, 0.95 * tau)",
            "exposure_window_frames": EXPOSURE_WINDOW_FRAMES,
            "packet_frames": 8_000_000,
            "top_k_recipients": TOP_K_RECIPIENTS,
            "permuted_route": arm == "MSA_PERMUTED_ROUTE",
            "permuted_route_seed": PERMUTED_ROUTE_SEED,
            "eval_num_seeds": FORMAL_NUM_SEEDS,
            "eval_seed_manifest_id": manifest_id,
            "eval_seed_first": seeds[0],
            "eval_seed_last": seeds[-1],
            "bootstrap": {"seed": BOOTSTRAP_SEED, "repeats": BOOTSTRAP_REPEATS,
                          "quantile": 0.05,
                          "statistic": "panel median of per-worker L_mid"},
            "rho_threshold": MSA_RHO_THRESHOLD,
            "rho_rule": "p_post/p_pre at the SAME tau; p_pre>=0.01 required; "
                        "cross-tolerance historical peaks forbidden",
            "numeric_gates": "unchanged from base-retention; only the "
                             "cross-tolerance retention read is replaced by "
                             "same-tau rho",
        }
    return cfg


_LAUNCHER_TEMPLATE = """#!/usr/bin/env bash
# {arm} continuation arm of base_retention_msa_probe.
# GENERATED by dexpbt_isaaclab.msa.arms — DO NOT LAUNCH without the
# LAUNCH_CHECKLIST sign-off. This arm resumes all six workers from the 5B
# tip checkpoints (per-policy warm start) with matched seeds.
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

RECOVER_LIST="{recover_list}"

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
  --extra \\
  --seed "$BASE_SEED" \\
{base_flags}  >>"$LOG" 2>&1 &

echo $! > "$RUN_DIR/nohup.pid"
{sidecar}echo "launched {arm}: run_dir=$RUN_DIR"
"""

_SIDECAR_BLOCK = """
# MSA probe sidecar: drives the probe state machine over the shared
# curriculum_control chain (reads tick metas, forces tau_plus/tau via
# forced_success_tolerance, runs policy-only evals in subprocesses).
nohup python -u -m dexpbt_isaaclab.msa.sidecar \\
  --workspace "$RUN_DIR/dexpbt_8gpu_ws" \\
  --arm {arm} \\
  --rho-threshold {rho} \\
  {permuted_flag}>>"$RUN_DIR/msa_sidecar.log" 2>&1 &
echo $! > "$RUN_DIR/msa_sidecar.pid"
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
    if arm == "GATE_RELAX":
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


def write_arm_launchers(out_dir: Path, manifest_csv: Path,
                        original_launcher: Path) -> list[Path]:
    """Write <arm>.sh + <arm>.config.json for all four arms. Never executes."""
    _validate_gate_relax_profile()
    tips = resolve_tip_checkpoints(manifest_csv)
    recover_list = ",".join(t.checkpoint_path for t in tips)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for arm in ARM_NAMES:
        cfg = arm_config(arm, tips)
        cfg_path = out_dir / f"{arm.lower()}.config.json"
        cfg_path.write_text(json.dumps(cfg, indent=2))
        written.append(cfg_path)
        sidecar = ""
        if arm in ("MSA", "MSA_PERMUTED_ROUTE"):
            sidecar = _SIDECAR_BLOCK.format(
                arm=arm, rho=MSA_RHO_THRESHOLD,
                permuted_flag=("--permuted-route "
                               if arm == "MSA_PERMUTED_ROUTE" else ""))
        wm_profile = GATE_RELAX_PROFILE if arm == "GATE_RELAX" else BASE_PROFILE
        script = _LAUNCHER_TEMPLATE.format(
            arm=arm, arm_lower=arm.lower(), recover_list=recover_list,
            max_iter=CONTINUATION_MAX_ITER,
            profile_list=",".join([wm_profile] * 6),
            base_flags=_base_flags_block(arm, original_launcher),
            sidecar=sidecar)
        sh_path = out_dir / f"start_{arm.lower()}.sh"
        sh_path.write_text(script)
        sh_path.chmod(0o755)
        written.append(sh_path)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--original-launcher", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    written = write_arm_launchers(Path(args.out_dir), Path(args.manifest_csv),
                                  Path(args.original_launcher))
    for p in written:
        print(p)
    print("NOTE: launchers were generated but NOT executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
