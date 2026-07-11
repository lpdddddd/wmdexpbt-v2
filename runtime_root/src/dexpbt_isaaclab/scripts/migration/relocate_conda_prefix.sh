#!/usr/bin/env bash
# Rewrite text files that embed the old conda env prefix (tar migration, no conda-pack).
set -euo pipefail
OLD="${1:?old prefix}"
NEW="${2:?new prefix}"
[[ -d "$NEW" ]] || { echo "missing env dir: $NEW" >&2; exit 1; }

log() { echo "[relocate] $*"; }

log "scanning under $NEW"
mapfile -t files < <(grep -rIl "$OLD" "$NEW" 2>/dev/null || true)
for f in "${files[@]}"; do
  if file -b "$f" | grep -qiE 'text|script|ascii'; then
    sed -i "s|${OLD}|${NEW}|g" "$f"
  fi
done
log "updated ${#files[@]} paths (text files only; binaries may still need conda-unpack)"
