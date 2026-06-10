#!/usr/bin/env bash
# Compact a finished DexPBT run by removing intermediate checkpoints.
#
# Usage:
#   bash cleanup_run.sh RUN_DIR [KEEP_TOP=3] [--dry-run]
#
# Keeps:
#   - newest wm_iter*.pt per policy (drops every earlier snapshot)
#   - top-K best_nsucc / best / best_r_meta / best_stable entries (json+pth pairs)
#   - newest dexpbt_8gpu_ws/00X/NNNNNN.pth per policy (PBT tick snapshots)
#   - last_*ep_<final>*.pth per policy_NN/rl_games/*/nn (final epoch only)
#   - all metrics, curriculum_control, training_curve, tfevents, launch_audit
set -euo pipefail

RUN_DIR="${1:?usage: $0 RUN_DIR [KEEP_TOP=3] [--dry-run]}"
KEEP="${2:-3}"
DRY=""
for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && DRY="echo DRY-RUN: rm"
done
RM="${DRY:-rm -f}"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "error: RUN_DIR not found: $RUN_DIR" >&2
  exit 1
fi

WS="$RUN_DIR/dexpbt_8gpu_ws"
echo "[cleanup] RUN_DIR=$RUN_DIR keep_top=$KEEP dry_run=${DRY:+yes}"
echo "[cleanup] before: $(du -sh "$RUN_DIR" | cut -f1)"

# 1) wm_checkpoints — keep only the newest wm_iter*.pt per policy.
for d in "$WS"/[0-9][0-9][0-9]/wm_checkpoints; do
  [[ -d "$d" ]] || continue
  # ls -t orders newest first; keep [0], drop [1:]
  mapfile -t pts < <(ls -t "$d"/wm_iter*.pt 2>/dev/null || true)
  if [[ "${#pts[@]}" -gt 1 ]]; then
    for f in "${pts[@]:1}"; do $RM "$f"; done
    echo "[wm_checkpoints] $d: kept 1, removed $((${#pts[@]} - 1))"
  fi
done

# 2) best_* archives — keep top-K entries by sorted basename (highest iter first).
for kind in best_nsucc best best_r_meta best_stable; do
  d="$WS/$kind"
  [[ -d "$d" ]] || continue
  mapfile -t jsons < <(ls -1 "$d"/*.json 2>/dev/null | sort -r || true)
  if [[ "${#jsons[@]}" -gt "$KEEP" ]]; then
    for j in "${jsons[@]:$KEEP}"; do
      pth="${j%.json}.pth"
      $RM "$j"
      [[ -f "$pth" ]] && $RM "$pth"
    done
    echo "[$kind] kept $KEEP, removed $((${#jsons[@]} - KEEP))"
  fi
done

# 3) per-policy PBT-tick snapshots (dexpbt_8gpu_ws/00X/NNNNNN.{pth,json}).
#    Keep only the newest tick per policy.
for d in "$WS"/[0-9][0-9][0-9]; do
  [[ -d "$d" ]] || continue
  mapfile -t pths < <(ls -1 "$d"/[0-9][0-9][0-9][0-9][0-9][0-9].pth 2>/dev/null | sort -r || true)
  if [[ "${#pths[@]}" -gt 1 ]]; then
    for p in "${pths[@]:1}"; do
      json="${p%.pth}.json"
      $RM "$p"
      [[ -f "$json" ]] && $RM "$json"
    done
    echo "[$(basename "$d")] kept 1 tick snapshot, removed $((${#pths[@]} - 1))"
  fi
done

# 4) rl_games per-policy intermediate ckpts — keep only the highest-epoch last_*.pth.
for nn in "$RUN_DIR"/policy_*/rl_games/*/nn; do
  [[ -d "$nn" ]] || continue
  # Identify the final-epoch checkpoint (highest ep number).
  final="$(ls -1 "$nn"/last_*ep_*.pth 2>/dev/null | sed -E 's/.*ep_([0-9]+).*/\1\t&/' | sort -k1,1 -n | tail -1 | cut -f2- || true)"
  if [[ -n "$final" ]]; then
    for f in "$nn"/last_*ep_*.pth; do
      [[ -f "$f" ]] || continue
      [[ "$f" == "$final" ]] && continue
      $RM "$f"
    done
  fi
  # rl-games auto-resume copy is redundant with the final last_*.pth.
  for f in "$nn"/DexPBTKukaAllegroLSTMPPO.pth; do
    [[ -f "$f" ]] || continue
    $RM "$f"
  done
done

echo "[cleanup] after:  $(du -sh "$RUN_DIR" | cut -f1)"
