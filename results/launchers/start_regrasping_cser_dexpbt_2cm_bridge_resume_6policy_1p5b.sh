#!/usr/bin/env bash
# 6-policy population continuation under cser_dexpbt_2cm_bridge.
#
# Forks the source population (policy i <- source policy i), selects a strong
# near-2cm checkpoint per worker, restores seeds/tolerances, and continues for
# ADDITIONAL_FRAMES with target_tolerance=TARGET_TOL.
set -euo pipefail

RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
REPO_ROOT="/root/autodl-tmp/wmdexpbt"
RUNS_ROOT="${RESULTS_ROOT}/runs"
SRC_RUN="${SRC_RUN:-${RUNS_ROOT}/regrasping_cser_dexpbt_conservative_retention_strict_5b_20260709_193412}"
ADDITIONAL_FRAMES="${ADDITIONAL_FRAMES:-1500000000}"
TARGET_TOL="${TARGET_TOL:-0.018}"
DEXPBT_GPUS="${DEXPBT_GPUS:-0,1,2,3,4,5}"
MAX_ITER="${MAX_ITER:-90000}"
NUM_POLICIES="${NUM_POLICIES:-6}"
BASE_SEED="${BASE_SEED:-}"
PBT_START_AFTER="${PBT_START_AFTER:-200000000}"
PBT_INITIAL_DELAY="${PBT_INITIAL_DELAY:-50000000}"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUN_DIR:-${RUNS_ROOT}/cont6_cser_2cm_bridge_1p5b_${DATE}}"
LOG="${RUN_DIR}/launch_master.log"
PIDFILE="${RUN_DIR}/nohup.pid"
COMMIT_SHA="$(cd "$REPO_ROOT" && git rev-parse HEAD)"
BUILD_MANIFEST="${RESULTS_ROOT}/launchers/build_cont6_resume_manifest.py"
MONITOR_SRC="${RESULTS_ROOT}/launchers/monitor_live_bridge.py"

SHARED_WM_HEAD_FLAGS=(
  --seq_wm_head_balance_enable
  --seq_wm_head_pos_weight_auto
  --seq_wm_head_pos_weight_max 10.0
  --seq_wm_lambda_horizon_success 0.5
  --seq_wm_lambda_drop_risk 0.5
  --seq_wm_lambda_recovery 0.5
  --seq_wm_replay_prioritize_floor_recovery
  --seq_wm_floor_aware_weighting
  --eir_candidate_gate_mode profile_soft
  --eir_candidate_min_weight_for_retrieval 0.05
)

PROFILE_LIST="cser_dexpbt_2cm_bridge,cser_dexpbt_2cm_bridge,cser_dexpbt_2cm_bridge,cser_dexpbt_2cm_bridge,cser_dexpbt_2cm_bridge,cser_dexpbt_2cm_bridge"

export DEXPBT_MAX_ACTIVE="${DEXPBT_MAX_ACTIVE:-6}"
export DEXPBT_INTER_LAUNCH_DELAY_S="${DEXPBT_INTER_LAUNCH_DELAY_S:-180}"
export DEXPBT_STARTUP_TIMEOUT_S="${DEXPBT_STARTUP_TIMEOUT_S:-1200}"
export DEXPBT_STARTUP_MAX_RESTARTS="${DEXPBT_STARTUP_MAX_RESTARTS:-3}"
export DEXPBT_NUM_POLICIES="$NUM_POLICIES"
export NUM_POLICIES
export DEXPBT_GPUS

if [[ ! -d "$SRC_RUN" ]]; then
  echo "[cont6] FATAL: SRC_RUN missing: $SRC_RUN" >&2
  exit 2
fi

# Refuse to launch if target GPUs are busy.
python3 - "$DEXPBT_GPUS" <<'PY'
import sys
gpus = [g.strip() for g in sys.argv[1].split(",") if g.strip()]
try:
    import subprocess
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
except Exception as exc:
    print(f"[cont6] FATAL: nvidia-smi failed: {exc}", file=sys.stderr)
    sys.exit(3)
busy = []
for line in out.splitlines():
    idx_s, mem_s = [x.strip() for x in line.split(",")]
    if idx_s in gpus and float(mem_s) > 200:
        busy.append((idx_s, mem_s))
if busy:
    print(f"[cont6] FATAL: GPUs not idle: {busy}", file=sys.stderr)
    sys.exit(4)
print("[cont6] GPUs idle OK")
PY

# Preflight tests (must pass).
cd "${RUNTIME_ROOT}/src/dexpbt_isaaclab"
python3 - <<'PY'
import sys, types
if "rl_games.common.algo_observer" not in sys.modules:
    m = types.ModuleType("rl_games")
    c = types.ModuleType("rl_games.common")
    a = types.ModuleType("rl_games.common.algo_observer")
    class IsaacAlgoObserver:
        def __init__(self, *a, **k): pass
    a.IsaacAlgoObserver = IsaacAlgoObserver
    sys.modules["rl_games"] = m
    sys.modules["rl_games.common"] = c
    sys.modules["rl_games.common.algo_observer"] = a
import pytest
rc = pytest.main(["-q", "tests/test_2cm_bridge_runtime_fixes.py"])
raise SystemExit(rc)
PY
python -m compileall dexpbt_isaaclab >/dev/null

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "${RESULTS_ROOT}/pointers/latest_cont6_cser_2cm_bridge_1p5b_run.txt"
echo "$RUN_DIR" > "${RUN_DIR}/RUN_DIR.txt"
cp -f "$MONITOR_SRC" "${RUN_DIR}/monitor_live_bridge.py"

SEED_ARGS=()
if [[ -n "$BASE_SEED" ]]; then
  SEED_ARGS+=(--base-seed "$BASE_SEED")
fi

echo "[cont6] building resume manifest + forking workspace..."
python3 "$BUILD_MANIFEST" \
  --src-run "$SRC_RUN" \
  --dst-run "$RUN_DIR" \
  --additional-frames "$ADDITIONAL_FRAMES" \
  --target-tol "$TARGET_TOL" \
  --num-policies "$NUM_POLICIES" \
  "${SEED_ARGS[@]}"

MANIFEST="${RUN_DIR}/resume_manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
  echo "[cont6] FATAL: resume_manifest.json missing" >&2
  exit 5
fi

# Sanity: every checkpoint exists; target = source + additional.
python3 - "$MANIFEST" "$ADDITIONAL_FRAMES" <<'PY'
import json, sys
from pathlib import Path
man = json.loads(Path(sys.argv[1]).read_text())
add = int(sys.argv[2])
for p in man["policies"]:
    ckpt = Path(p["checkpoint"])
    if not ckpt.is_file():
        raise SystemExit(f"missing checkpoint policy {p['policy_idx']}: {ckpt}")
    src = int(p["source_population_frame"])
    tgt = int(p["target_population_frames"])
    if tgt != src + add:
        raise SystemExit(
            f"target mismatch policy {p['policy_idx']}: {tgt} != {src}+{add}"
        )
    if float(p["success_tolerance_resume"]) <= 0:
        raise SystemExit(f"bad tolerance policy {p['policy_idx']}")
print("[cont6] manifest sanity OK")
print(f"[cont6] base_seed={man['base_seed']} seed_rule={man['seed_rule']}")
for p in man["policies"]:
    print(
        f"  p{p['policy_idx']}: iter={p['source_iteration']} "
        f"frame={p['source_population_frame']}->{p['target_population_frames']} "
        f"tol={p['success_tolerance_resume']:.6f} "
        f"mean={p['source_nsucc_mean']:.2f} "
        f"seed={p['seed']}/{p['effective_seed']} "
        f"warn={p['source_checkpoint_quality_warning']}"
    )
PY

# Ensure new workspace is not a symlink into the source run.
python3 - "$SRC_RUN" "$RUN_DIR" <<'PY'
import sys
from pathlib import Path
src = Path(sys.argv[1]).resolve()
dst = Path(sys.argv[2]).resolve()
ws = dst / "dexpbt_8gpu_ws"
ctrl = ws / "curriculum_control"
if not ctrl.is_dir():
    raise SystemExit("forked curriculum_control missing")
# Mutable control files must be real copies (not hardlinks to source).
for p in ctrl.glob("*.json"):
    try:
        if p.samefile(src / "dexpbt_8gpu_ws" / "curriculum_control" / p.name):
            raise SystemExit(f"control still linked to source: {p}")
    except FileNotFoundError:
        pass
print("[cont6] workspace fork isolation OK")
PY

cat > "${RUN_DIR}/RUN_CONFIG.txt" <<EOF
launcher=start_regrasping_cser_dexpbt_2cm_bridge_resume_6policy_1p5b.sh
method=cser_dexpbt_2cm_bridge
mode=population_continuation_6policy
config_version=cser_dexpbt_2cm_bridge_cont6_1p5b
commit=${COMMIT_SHA}
source_run=${SRC_RUN}
additional_frames=${ADDITIONAL_FRAMES}
target_tolerance=${TARGET_TOL}
termination_mode=per_worker_population_frames
max_iterations=${MAX_ITER}
max_iterations_role=safety_cap_only
max_active=6
num_policies=${NUM_POLICIES}
gpus=${DEXPBT_GPUS}
profile_all_workers=cser_dexpbt_2cm_bridge
profile_copy=false
awr=false
eir_residual_rollout_apply=false
pbt_allow_mid_self_mutation=false
mid_self_mutation_disabled=true
resume_manifest=${MANIFEST}
identity_note=new policy i continues source policy i (same policy_idx, same base seed rule)
bridge_additions=same_tol_retention,bridge_ladder,bridge_probation,snippet_adjacent_compat,late_degradation_rescue,frame_stop
consolidation_note=absolute population frames continue; stage3_end kept at 5B so frames>=5B remain STAGE_SELECTION (mutation 0, bottom hard-collapse / late-rescue still allowed); bridge curriculum continues toward 0.018
note=6-policy directed continuation from near-2cm checkpoints; source run is read-only.
EOF

source "${RUNTIME_ROOT}/scripts/env_dexpbt.sh"
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
export OMNI_KIT_TELEMETRY=0
export CARB_CRASH_REPORTER=0

cd "${RUNTIME_ROOT}/src/dexpbt_isaaclab"

# Shared --extra must NOT set a single target_population_frames / seed /
# success_tolerance_resume; those come from --resume-manifest per policy.
nohup bash "${RUNTIME_ROOT}/scripts/launch_8gpu_dexpbt.sh" \
  "$RUN_DIR" \
  --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \
  --num-envs 8192 \
  --max-iterations "$MAX_ITER" \
  --pbt-interval-steps 20000000 \
  --pbt-start-after "$PBT_START_AFTER" \
  --pbt-initial-delay "$PBT_INITIAL_DELAY" \
  --profile-per-policy "$PROFILE_LIST" \
  --resume-manifest "$MANIFEST" \
  --extra \
  --pbt_ranking_metric success_aware_r_meta \
  --pbt_min_frames_as_source_after_restart 50000000 \
  --pbt_min_frames_between_restarts 100000000 \
  --pbt_disable_self_restart \
  --no-pbt_allow_mid_self_mutation \
  --pbt_source_selection_mode tiered \
  --no-pbt_enable_best_nsucc_recovery \
  --pbt_allow_best_nsucc_archive_recovery \
  --pbt_recovery_cooldown_frames 400000000 \
  --pbt_curriculum_freeze_frames 100000000 \
  --pbt_curriculum_freeze_on_recovery_frames 100000000 \
  --curriculum_mode mastery \
  --pbt_source_require_stable_mean \
  --mastery_nsucc_mean 4.0 \
  --mastery_nsucc_max 20.0 \
  --mastery_consecutive_windows 2 \
  --curriculum_shrink_factor 0.95 \
  --pbt_single_shrink_per_window \
  --post_shrink_consolidation_env_frames 150000000 \
  --curriculum_speed_mode adaptive_fast \
  --curriculum_fast_strong_mean 30.0 \
  --curriculum_fast_mid_mean 20.0 \
  --curriculum_fast_low_mean 10.0 \
  --curriculum_fast_retention_threshold 0.8 \
  --curriculum_fast_retention_window 4 \
  --curriculum_fast_shrink_strong 0.925 \
  --curriculum_fast_shrink_mid 0.95 \
  --curriculum_fast_shrink_low 0.975 \
  --curriculum_fast_consolidation_frames 100000000 \
  --curriculum_fast_recovery_consolidation_frames 100000000 \
  --curriculum_fast_target_tolerance_800m 0.045 \
  --pbt_curriculum_inherit_mode source_safe \
  --pbt_source_safe_recovery_consolidation_frames 100000000 \
  --pbt_source_safe_min_mean 20.0 \
  --pbt_source_safe_min_retention 0.7 \
  --pbt_protect_promising_workers \
  --pbt_promising_nsucc_mean 20.0 \
  --pbt_promising_peak_mean 25.0 \
  --pbt_promising_min_retention 0.5 \
  --pbt_promising_protect_frames 250000000 \
  --pbt_promising_allow_hard_collapse_override \
  --pbt_strong_live_inherit_exact_enable \
  --pbt_strong_live_inherit_max_tol 0.035 \
  --pbt_strong_live_inherit_min_mean 30.0 \
  --pbt_strong_live_inherit_min_max 35.0 \
  --pre_floor_peak_guard_enable \
  --pre_floor_peak_guard_tol_threshold 0.030 \
  --pre_floor_peak_guard_target_tol 0.025 \
  --pre_floor_peak_guard_peak_threshold 35.0 \
  --pre_floor_peak_guard_min_retention 0.92 \
  --pre_floor_peak_guard_min_stable_ticks 3 \
  --pre_floor_peak_guard_freeze_frames 200000000 \
  --late_floor_retention_gate_enable \
  --late_floor_retention_tol_threshold 0.030 \
  --late_floor_retention_peak_threshold 35.0 \
  --late_floor_retention_min_retention 0.88 \
  --late_floor_retention_hard_min_retention 0.75 \
  --late_floor_retention_patience 2 \
  --late_floor_retention_freeze_frames 250000000 \
  --late_floor_retention_block_shrink \
  --late_floor_retention_freeze_mutation \
  --no-late_floor_retention_rollback_enable \
  --pbt_floor_recovery_aware \
  --pbt_floor_tol_threshold 0.030 \
  --pbt_hard_floor_tol_threshold 0.025 \
  --pbt_floor_recovery_window 4 \
  --pbt_floor_recovery_protect_frames 300000000 \
  --pbt_floor_recovery_min_nsucc_protect 15.0 \
  --pbt_enable_best_floor_recovery_archive \
  --no-pbt_wm_profile_copy_enable \
  --no-awr_enable \
  --no-eir_residual_apply_to_rollout \
  --no-eir_use_rssm_mppi \
  --eir_mppi_mode off \
  --elite_snippet_enable \
  --snippet_distill_enable \
  --snippet_wm_rerank_enable \
  --wm_calibration_enable \
  --wm_calibration_eval_tol_mode hybrid \
  --wm_calibration_eval_tol_threshold 0.05 \
  --wm_calibration_eval_tol_cap 0.06 \
  --wm_calibration_min_samples 128 \
  --eir_actor_distill_enable \
  --eir_actor_distill_allowed_sources elite,archive \
  --eir_actor_distill_weight 0.0005 \
  --eir_actor_distill_loss_cap 0.01 \
  --eir_actor_distill_max_delta 0.25 \
  --actor_distill_fallback_with_snippet_enable \
  --actor_distill_fallback_min_snippet_bank_size 5000 \
  --actor_distill_fallback_min_rerank_changed_frac 0.0001 \
  --pbt_consolidation_enable \
  --pbt_consolidation_stage1_end_frames 2200000000 \
  --pbt_consolidation_stage2_end_frames 3500000000 \
  --pbt_consolidation_stage3_end_frames 5000000000 \
  --pbt_consolidation_stage2_interval_mult 2.0 \
  --pbt_consolidation_stage3_interval_mult 4.0 \
  --pbt_consolidation_stage2_mutation_rate_mult 0.5 \
  --pbt_consolidation_stage3_bottom_only \
  --pbt_consolidation_stage3_hard_collapse_only \
  --pbt_consolidation_stage2_retention_threshold 0.85 \
  --pbt_consolidation_collapse_nsucc_mean_threshold 1.0 \
  --pbt_early_collapse_detect \
  --early_collapse_drop_ratio 0.3 \
  --early_collapse_patience 1 \
  --curriculum_collapse_detect \
  --collapse_drop_ratio 0.5 \
  --collapse_patience 2 \
  --collapse_freeze_env_frames 200000000 \
  --stable_source_min_mean 3.0 \
  --stable_source_min_max 20.0 \
  --pbt_recovery_bypass_elimination_burnin \
  --pbt_recovery_tolerance_mode last_mastered \
  --pbt_crosscopy_tolerance_mode last_mastered \
  --curriculum_rollback_cooldown_frames 50000000 \
  --curriculum_safety_enable \
  --curriculum_safety_start_after_frames 500000000 \
  --curriculum_safety_min_nsucc_mean 10.0 \
  --curriculum_safety_global_retention_threshold 0.80 \
  --curriculum_safety_drop_ratio_threshold 0.30 \
  --curriculum_safety_slope_threshold -2.0 \
  --curriculum_safety_window 6 \
  --curriculum_safety_block_frames 100000000 \
  --curriculum_safety_use_real_fallback \
  --curriculum_strong_shrink_global_retention_gate \
  --curriculum_strong_shrink_min_global_retention 0.85 \
  --curriculum_strong_shrink_min_stable_ticks 3 \
  --curriculum_strong_shrink_long_window 12 \
  --curriculum_danger_zone_enable \
  --curriculum_danger_tol 0.030 \
  --curriculum_danger_disable_strong_shrink \
  --curriculum_danger_min_stable_ticks 4 \
  --curriculum_danger_shrink_interval_mult 2.0 \
  --curriculum_danger_global_retention_threshold 0.85 \
  --curriculum_rollback_on_repeated_collapse \
  --curriculum_rollback_collapse_patience 2 \
  --curriculum_rollback_to_last_mastered \
  --curriculum_rollback_freeze_frames 200000000 \
  --curriculum_last_mastered_min_nsucc_mean 30.0 \
  --curriculum_last_mastered_min_global_retention 0.85 \
  --curriculum_last_mastered_stable_ticks 3 \
  --wm_imagined_diagnostic_enable \
  --wm_imagined_shadow_block_enable \
  --wm_imagined_shadow_uncertainty_threshold 0.95 \
  --frame_stop_enable \
  --frame_stop_grace_iterations 2 \
  --same_tol_retention_enable \
  --bridge_curriculum_enable \
  --bridge_probation_enable \
  --snippet_adjacent_stage_mixing_enable \
  --snippet_tolerance_compatibility_enable \
  --late_degradation_rescue_enable \
  --late_degradation_rescue_source_min_mean 28.0 \
  --late_degradation_rescue_min_source_advantage 8.0 \
  --late_degradation_rescue_source_min_retention 0.85 \
  --late_degradation_rescue_reject_collapse_source \
  --late_degradation_rescue_require_source_control \
  --no-late_degradation_rescue_bypass_elimination_burnin \
  "${SHARED_WM_HEAD_FLAGS[@]}" \
  >>"$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "[cont6] launch_master pid=$(cat "$PIDFILE") log=$LOG run_dir=$RUN_DIR"

# Start live monitor (detached).
nohup python3 -u "${RUN_DIR}/monitor_live_bridge.py" \
  --run-dir "$RUN_DIR" \
  --interval-s 60 \
  --num-policies "$NUM_POLICIES" \
  >>"${RUN_DIR}/monitor_live_bridge.log" 2>&1 &
echo $! > "${RUN_DIR}/monitor.pid"
echo "[cont6] monitor pid=$(cat "${RUN_DIR}/monitor.pid")"

echo "$RUN_DIR"
