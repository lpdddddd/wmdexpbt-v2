# Snapshot Validation Cache

Date: 2026-07-14  
Parent: `358133f`  
Commit: (see current HEAD after Optimize ABC snapshot validation caching)

## Problem

Chronic source selection hashed peer commit model/meta and published checkpoints
for every peer, then post-selection defense-in-depth repeated the same full
SHA-256 on the selected source.

## Design

Process-local cache in `PeerSnapshotHealthCache`
(`pbt/pbt_live_source_health.py`), owned by each observer instance.

### Fingerprint (no Python `hash()`)

`SnapshotValidationFingerprint` includes policy_idx, commit_id, commit_dir,
LATEST/MANIFEST/checkpoint size+mtime_ns, commit model/meta size+mtime_ns,
state lineage_id, state population_frame, tick.

### Semantics

1. Build lightweight fingerprint first (stat + small JSON reads).
2. Exact fingerprint match → cache hit; return deep-copied `PeerCheckpointHealth`;
   **no** `validate_snapshot_commit`, **no** checkpoint SHA-256.
3. Miss → full validation; store result (positive or negative) under fingerprint.
4. Cap: max 3 entries per policy (LRU).
5. Corrupt LATEST / unreadable fingerprint → always full inspect; never reuse
   a prior entry for that call.
6. Skill_delta / probation / collapse / ranking are **not** cached.
7. `population_frame_complete` still unused for freshness.

### Call sites

- Selection: `_evaluate_common_live_source_health` → `_inspect_peer_checkpoint_health`
- Post-selection: same path; fingerprint rebuilt; hit when unchanged.

### Telemetry

TB: `pbt/chronic_snapshot_cache_{hit,miss,hit_count,miss_count,entries}`,
`pbt/chronic_snapshot_validation_{latency,full_hash}`

JSONL fields: `cache_hit`, `cache_key_commit_id`, `cache_key_commit_dir`,
`validation_latency_s`, `full_hash_executed`, `cache_invalidated_reason`
(`first_validation|latest_changed|checkpoint_stat_changed|manifest_changed|
lineage_changed|frame_changed|tick_changed|cache_hit|…`).

## Test evidence

`TestSnapshotValidationCache`:

| Case | Result |
|---|---|
| First legal snapshot | miss + full hash + pass |
| Second identical | hit; validate_snapshot_commit count stays 1 |
| Selection → post-selection | hit |
| New LATEST commit | miss `latest_changed` |
| Checkpoint mtime change | miss |
| Checkpoint size change | miss + identity fail |
| MANIFEST mtime change | miss |
| Lineage / frame mismatch | miss + fail-close |
| Corrupt LATEST | fail-close; no stale hit |
| Tampered model | fail-close |
| Capacity | ≤2 with max_per_policy=2 |
| New cache object | process-local miss |

Validation call-count comparison (spy): first inspect 1× `validate_snapshot_commit`;
identical second inspect 0 additional calls.
