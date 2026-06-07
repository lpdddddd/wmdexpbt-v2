#!/usr/bin/env bash
# Pinned-checkpoint cap sweep wrapper.
#
# Runs eval_cap_sweep.py with --policy_checkpoint_iterations forwarded to
# eval_fair so the user-requested PBT ticks are evaluated instead of the
# r_meta-best ticks. Outputs land under <run-dir>/eval_cap_sweep_pinned/
# (and <run-dir>/EVAL_CAP_SWEEP_PINNED.json) so they do NOT collide with
# any prior --checkpoint_rule sweep.
#
# Defaults pin the fixed_h1 5B formal run. Override via env vars:
#   RUN_DIR        — absolute path to the run directory
#   PINS           — `0=108,1=137,2=131,3=135,4=83,5=114` style map
#   EXPECTED       — `0=2160000000,...` env_frames map (frame mismatch warning)
#   CAPS           — caps list passed to eval_cap_sweep, default `50 100 200`
#   EPISODES       — episodes per policy, default 256
#   NUM_ENVS       — eval num_envs, default 512
#   METHOD_NAME    — CSV tag, default `fixed_h1_pinned`
#   OUT_SUBDIR     — output subdir under RUN_DIR, default eval_cap_sweep_pinned
#   LOG            — eval stdout/stderr log (default <RUN_DIR>/eval_pinned.log)
#
# Plan reference: post-hoc_eval_+_track_c_bayesian-no-h3 §2.1.

set -euo pipefail

WMDEXPBT_ROOT="/root/autodl-tmp/wmdexpbt"

RUN_DIR="${RUN_DIR:-${WMDEXPBT_ROOT}/results/regrasping_fixed_h1_p2_1_formal_5b_20260605_224754}"
PINS="${PINS:-0=108,1=137,2=131,3=135,4=83,5=114}"
EXPECTED="${EXPECTED:-0=2160000000,1=2740000000,2=2620000000,3=2700000000,4=1660000000,5=2280000000}"
CAPS="${CAPS:-50 100 200}"
EPISODES="${EPISODES:-256}"
NUM_ENVS="${NUM_ENVS:-512}"
METHOD_NAME="${METHOD_NAME:-fixed_h1_pinned}"
OUT_SUBDIR="${OUT_SUBDIR:-eval_cap_sweep_pinned}"
LOG="${LOG:-${RUN_DIR}/eval_pinned.log}"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "[eval_pinned] run dir not found: ${RUN_DIR}" >&2
  exit 1
fi

# Source Isaac env so subprocess `python` is the IsaacLab one.
# shellcheck disable=SC1091
source "${WMDEXPBT_ROOT}/runtime_root/scripts/env_dexpbt.sh"

# Match training: one visible GPU avoids rtx.scenedb multi-GPU startup SIGSEGV.
export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"

ISAAC_PY="${WMDEXPBT_ROOT}/runtime_root/envs/isaaclab311/bin/python"
if [[ ! -x "${ISAAC_PY}" ]]; then
  echo "[eval_pinned] Isaac python not found at ${ISAAC_PY}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}/${OUT_SUBDIR}"

echo "[eval_pinned] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[eval_pinned] RUN_DIR=${RUN_DIR}"
echo "[eval_pinned] PINS=${PINS}"
echo "[eval_pinned] EXPECTED=${EXPECTED}"
echo "[eval_pinned] CAPS=${CAPS}"
echo "[eval_pinned] OUT_SUBDIR=${OUT_SUBDIR}"
echo "[eval_pinned] LOG=${LOG}"

# Run cap sweep — eval_cap_sweep.py forwards --extra-eval-args to
# eval_fair, including the per-policy iteration map and the expected
# frame map (mismatch >10M frames -> WARNING, eval continues).
nohup "${ISAAC_PY}" "${WMDEXPBT_ROOT}/results/eval_cap_sweep.py" \
  --run-dir "${RUN_DIR}" \
  --caps ${CAPS} \
  --episodes "${EPISODES}" \
  --num-envs "${NUM_ENVS}" \
  --method-name "${METHOD_NAME}" \
  --out-subdir "${OUT_SUBDIR}" \
  --extra-eval-args \
  --policy_indices 0,1,2,3,4,5 \
  --policy_checkpoint_iterations "${PINS}" \
  --policy_expected_frames "${EXPECTED}" \
  >"${LOG}" 2>&1 &
SWEEP_PID=$!
echo "[eval_pinned] eval_cap_sweep pid=${SWEEP_PID}"
echo "[eval_pinned] Tail log with: tail -f ${LOG}"
