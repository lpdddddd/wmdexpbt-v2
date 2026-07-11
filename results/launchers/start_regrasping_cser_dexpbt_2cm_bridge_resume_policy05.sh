#!/usr/bin/env bash
# Fork policy 05 from an existing strict-5B run at a chosen checkpoint iter and
# continue under the 2cm bridge stack.
#
# Semantics:
#   START_TOL  -> current success_tolerance (where weights currently sit)
#   TARGET_TOL -> curriculum target (bridge continues toward this)
#   CONTINUE_FRAMES -> additional population-frame budget
#   absolute target = SOURCE_POPULATION_FRAME + CONTINUE_FRAMES
set -euo pipefail

RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
SRC_RUN="${SRC_RUN:-${RESULTS_ROOT}/runs/regrasping_cser_dexpbt_conservative_retention_strict_5b_20260709_193412}"
SRC_WS="${SRC_RUN}/dexpbt_8gpu_ws"
SRC_POLICY_ID="${SRC_POLICY_ID:-5}"
SRC_ITER="${SRC_ITER:-131}"
SRC_CKPT="${SRC_CKPT:-${SRC_WS}/$(printf '%03d' "${SRC_POLICY_ID}")/$(printf '%06d' "${SRC_ITER}").pth}"
START_TOL="${START_TOL:-0.021}"
TARGET_TOL="${TARGET_TOL:-0.018}"
CONTINUE_FRAMES="${CONTINUE_FRAMES:-800000000}"
MAX_ITER="${MAX_ITER:-9000}"
LOCAL_POLICY_ID=0
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RESULTS_ROOT}/runs/cont_cser_2cm_bridge_p05_${DATE}"

if [[ ! -f "$SRC_CKPT" ]]; then
  echo "[resume] source checkpoint not found: $SRC_CKPT" >&2
  echo "[resume] provide SRC_RUN / SRC_ITER / SRC_CKPT env vars to point at a real checkpoint." >&2
  exit 2
fi

# Read authoritative population frame from checkpoint payload / sibling meta.
SOURCE_POPULATION_FRAME="$(python3 - "$SRC_CKPT" <<'PY'
import json, sys
from pathlib import Path
ckpt = Path(sys.argv[1])
candidates = [
    ckpt.with_suffix(".json"),
    ckpt.parent / f"{ckpt.stem}_meta.json",
    ckpt.parent / "checkpoint_meta.json",
]
frame = None
# Prefer torch payload fields when available.
try:
    import torch
    payload = torch.load(str(ckpt), map_location="cpu")
    if isinstance(payload, dict):
        for key in (
            "population_frame",
            "pbt_population_frame",
            "frame",
            "env_frames",
        ):
            if key in payload and payload[key] is not None:
                frame = int(payload[key])
                break
        cur = payload.get("curriculum") if isinstance(payload.get("curriculum"), dict) else {}
        if frame is None and "population_frame" in cur:
            frame = int(cur["population_frame"])
except Exception as exc:
    print(f"[resume] torch payload read failed: {exc}", file=sys.stderr)

if frame is None:
    for path in candidates:
        if not path.exists():
            continue
        try:
            meta = json.loads(path.read_text())
        except Exception:
            continue
        for key in ("population_frame", "pbt_population_frame", "frame", "env_frames"):
            if key in meta and meta[key] is not None:
                frame = int(meta[key])
                break
        if frame is not None:
            break

if frame is None or frame < 0:
    print(
        "[resume] FATAL: could not read SOURCE_POPULATION_FRAME from checkpoint "
        f"or meta beside {ckpt}. Refusing to default to 0 / CONTINUE_FRAMES.",
        file=sys.stderr,
    )
    sys.exit(3)
print(frame)
PY
)"

TARGET_POPULATION_FRAMES="$(python3 -c "print(int(${SOURCE_POPULATION_FRAME}) + int(${CONTINUE_FRAMES}))")"

mkdir -p "$RUN_DIR"
cat > "$RUN_DIR/RUN_CONFIG.txt" <<EOF
launcher=$(basename "$0")
method=cser_dexpbt_2cm_bridge
mode=continuation_policy05
source_run=${SRC_RUN}
source_checkpoint=${SRC_CKPT}
source_policy_id=${SRC_POLICY_ID}
continuation_local_policy_id=${LOCAL_POLICY_ID}
source_iteration=${SRC_ITER}
source_population_frame=${SOURCE_POPULATION_FRAME}
current_tolerance=${START_TOL}
target_tolerance=${TARGET_TOL}
additional_frames=${CONTINUE_FRAMES}
absolute_target_population_frames=${TARGET_POPULATION_FRAMES}
termination_mode=per_worker_population_frames
max_iterations=${MAX_ITER}
max_iterations_role=safety_cap_only
identity_note=local policy ${LOCAL_POLICY_ID} is a continuation of source policy ${SRC_POLICY_ID}; do not confuse with original population policy 0
EOF

source "${RUNTIME_ROOT}/scripts/env_dexpbt.sh"
cd "${RUNTIME_ROOT}/src/dexpbt_isaaclab"
nohup python -u -m dexpbt_isaaclab.scripts.train \
  --task DexPBT-Regrasping-KukaAllegro-Direct-v0 --headless \
  --num_envs 8192 --max_iterations "$MAX_ITER" \
  --observer_mode pbt --policy_idx "$LOCAL_POLICY_ID" --num_policies 1 \
  --workspace "$RUN_DIR/dexpbt_8gpu_ws" \
  --wm_strategy_profile cser_dexpbt_2cm_bridge \
  --checkpoint "$SRC_CKPT" \
  --success_tolerance_resume "$START_TOL" \
  --target_tolerance_resume "$TARGET_TOL" \
  --frame_stop_enable --target_population_frames "$TARGET_POPULATION_FRAMES" \
  --same_tol_retention_enable --same_tol_retention_reset_on_cross_copy \
  --bridge_curriculum_enable --bridge_probation_enable \
  --snippet_adjacent_stage_mixing_enable --snippet_tolerance_compatibility_enable \
  --late_degradation_rescue_enable \
  --late_degradation_rescue_source_min_mean 28.0 \
  --late_degradation_rescue_min_source_advantage 8.0 \
  --late_degradation_rescue_source_min_retention 0.85 \
  --late_degradation_rescue_reject_collapse_source \
  --late_degradation_rescue_require_source_control \
  --no-late_degradation_rescue_bypass_elimination_burnin \
  --no-pbt_allow_mid_self_mutation \
  --no-pbt_wm_profile_copy_enable \
  --no-awr_enable \
  --no-eir_residual_apply_to_rollout \
  --no-eir_use_rssm_mppi \
  --eir_mppi_mode off \
  >"$RUN_DIR/launch_master.log" 2>&1 &
echo $! > "$RUN_DIR/nohup.pid"
echo "$RUN_DIR"
echo "[resume] source_policy=${SRC_POLICY_ID} local_policy=${LOCAL_POLICY_ID} start_tol=${START_TOL} target_tol=${TARGET_TOL}"
echo "[resume] source_frame=${SOURCE_POPULATION_FRAME} + continue=${CONTINUE_FRAMES} -> target=${TARGET_POPULATION_FRAMES}"
