# MSA_STATEFUL N=16 pilot — postmortem

Source run:
`/root/autodl-tmp/wmdexpbt/results/runs/base_retention_msa_probe_msa_stateful_20260720_033431`

Code at launch: `bf4d8b3` (lineage resume). Analysis artifacts live beside
this file under `results/analysis/msa_stateful_n16_20260720/`.

## 1. Evidence completeness

| Artifact | Status |
|---|---|
| FIRST_TICK_VALIDATED.json | copied |
| ARM_CONFIG / STATE_FIDELITY / STATE_FINGERPRINT | copied |
| GUARD_FAILED / MSA_SIDECAR_DIED / MSA_BUSY | copied |
| msa_cycles (cycle 1 + 2) | copied |
| snapshot `.meta.json` | copied (`.pth` **not** committed; sha/size in raw_manifest) |
| msa_eval_raw (672 files) | **not** copied; sha256+size manifest only |
| READINESS_HOLDS / MSA_HOLDS | **absent at stop** (cleared by stop_arm) |

Raw paths, sizes and hashes: `raw_manifest/`. Compact tables:
`CYCLE_TABLE.json`, `WORKER_TABLE.csv`, `FRAME_LEDGER.json`,
`SEED_LEVEL_EVIDENCE.json`, `SELECTIVE_REPLAY.json`.

## 2. What worked (orchestration / eval / lineage)

- State-faithful bootstrap: tolerance, env clock, lineage, observer seed,
  control restore.
- First-tick gate: 6/6 exact source τ, matching lineage, stall reproduced;
  sidecar adoption from readiness holds.
- Two full N=16 probe cycles with durable cycle JSON; panel bootstrap
  recomputed from eval raw (`normalized_nsucc`) **bit-matches** the
  on-disk `panel_stat.lower_quantile`.
- Fail-closed stop on cap overshoot and sidecar death mid-hold; GPUs freed.

## 3. Algorithmic non-promotion (not a bug)

### Cycle 1 — `msa_1784492852_5fa1236a` (due=[0..5])

- `promoted=false`, `first_blocking_gate=panel_lower_quantile`
- Panel L_mid 5% lower quantile ≈ **−0.071** (recomputed identical)
- Global mode correctly blocked the whole panel.
- **Worker-local selective replay** (same paired CIs + ρ≥0.88):
  - eligible: **w2, w3**
  - selected (top-3, ≥3 anchors): **w3, w2**
  - rejected: w0/w1/w4 (L_mid CI≤0); w0/w5 also ρ&lt;0.88

### Cycle 2 — `msa_1784527942_af1468d9` (due=[0,1,5])

- Panel gate **passed** (lower quantile ≈ **+0.160**)
- `promoted=false`, `first_blocking_gate=same_tau_retention_w5`
  (w5 ρ≈0.75; w0≈1.26; w1≈1.36)
- Global mode correctly vetoed everyone because of one weak worker.
- **Selective replay**:
  - eligible: **w0, w1** (L_mid lower bound &gt;0 and ρ≥0.88)
  - rejected: **w5** (`same_tau_retention`)
  - selected: **w0, w1**; anchors include w2–w5

## 4. Orchestration cap stop (bug)

### Cycle 3 — `msa_1784551259_a03ee1cf` (due=[2,3,4], PRE_EVAL)

- No cycle summary JSON (aborted before DECIDE).
- `GUARD_FAILED`: worker 2 frame **6.12B** &gt; baseline+480M+80M slack
  (**5.64B**).
- Guard logs: multiple workers `cap_pending` while `MSA_BUSY` held;
  after busy cleared, frames already past fail threshold.
- Watchdog: `MSA_SIDECAR_DIED` with active holds → released → stopped arm.

Root cause: **no worst-case dose reservation before opening a cycle**.
Busy only deferred park/fail; it did not prevent starting a cycle that
could not finish inside the remaining +480M budget.

## 5. What selective mode would have changed

| Cycle | Global (actual) | Selective (offline replay) |
|---|---|---|
| 1 | no promotion (panel) | promote **w3, w2** to τ⁺ |
| 2 | no promotion (w5 ρ) | promote **w0, w1**; reject w5 |

This is exactly the intended fix: strong workers are no longer vetoed by
a weak peer. Panel stats remain diagnostics under selective mode.

## 6. Limits (single training seed)

- One continuation training seed / one arm instance — no population CI
  over independent training runs.
- Bootstrap CIs are **conditional** on these checkpoints and the fixed
  N=16 eval seed manifest (pre-registered seed 20260719, 10k reps, 5th pct).
- ρ uses the existing point estimate gate (0.88); no separate ρ CI was
  persisted in the cycle JSON, so selective replay uses the same point
  gate (documented; not invented).
- Formal claims about MSA vs GATE_RELAX / BASE still require the full
  four-arm comparison; this pilot alone cannot settle that.
