#!/usr/bin/env bash
# Restart stopped CB-WM policies (default: only those with status=stopped).
# Usage: bash restart_cbwm_dead_policies.sh [RUN_DIR] [policy_ids...]
# Example: bash restart_cbwm_dead_policies.sh "$RUN" 4 5
set -euo pipefail

RUN="${1:-}"
if [[ -z "$RUN" && -f /root/autodl-tmp/wmdexpbt/results/latest_run.txt ]]; then
  RUN="$(tr -d '\n' < /root/autodl-tmp/wmdexpbt/results/latest_run.txt)"
fi
if [[ -z "$RUN" || ! -d "$RUN" ]]; then
  echo "usage: $0 RUN_DIR [policy_ids...]" >&2
  exit 1
fi
shift || true
if [[ $# -gt 0 ]]; then
  POLICIES=("$@")
else
  POLICIES=(0 1 2 3 4 5)
fi
RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
SUPERVISOR="${RUNTIME_ROOT}/src/dexpbt_isaaclab/run_train_supervisor.sh"
TASK="DexPBT-Regrasping-KukaAllegro-Direct-v0"
WS="${RUN}/dexpbt_8gpu_ws"
NUM_POLICIES="${DEXPBT_NUM_POLICIES:-6}"

source "${RUNTIME_ROOT}/scripts/env_dexpbt.sh"

# Only restart policies that are stopped (not actively training)
for i in "${POLICIES[@]}"; do
  pdir="${RUN}/policy_$(printf '%02d' $i)"
  st="${pdir}/status.txt"
  if [[ -f "$st" ]] && grep -q '^status=starting\|^status=restarting_pbt' "$st" 2>/dev/null; then
    echo "skip policy_${i}: still active"
    continue
  fi
  if [[ -f "$st" ]] && ! grep -q '^status=stopped' "$st" 2>/dev/null; then
    if pgrep -f "${pdir}.*run_train_supervisor" >/dev/null 2>&1; then
      echo "skip policy_${i}: supervisor running (status not stopped)"
      continue
    fi
  fi
  echo "restarting policy_${i} on GPU ${i}"
  export CUDA_VISIBLE_DEVICES="${i}"
  nohup bash "$SUPERVISOR" "$pdir" "$TASK" \
    --device=cuda:0 \
    --headless \
    --observer_mode=pbt \
    --policy_idx="${i}" \
    --num_policies="${NUM_POLICIES}" \
    --workspace="${WS}" \
    --pbt_interval_steps=20000000 \
    --pbt_start_after=200000000 \
    --pbt_initial_delay=50000000 \
    --pbt_mutation_rate=0.2 \
    --pbt_change_min=1.1 \
    --pbt_change_max=1.5 \
    --num_envs=8192 \
    --max_iterations=38147 \
    --enable_wm \
    --wm_strategy=value_imag_h1 \
    --wm_ensemble_size=3 \
    --wm_lr=1e-3 \
    --wm_update_every_n_epochs=2 \
    --wm_batch_size=512 \
    --enable_counterfactual \
    --cf_interval_gens=5 \
    >>"${pdir}/supervisor_restart.log" 2>&1 &
  echo "  pid=$!"
done

echo "Done. Monitor: tail -f ${RUN}/policy_00/train.log"
