# MSA GPU Execution Smoke — Run Report (2026-07-19)

Code under test: branch `base-retention`, commit `0ca91f2`
("Close MSA smoke, commit, and cap races"). No code was modified during
this run.

## Verdict

**FAIL-CLOSED AT READINESS STAGE.** The continuation guard rejected
worker 0's first committed tick and stopped the arm before
`FIRST_TICK_VALIDATED.json` was written. The hold smoke and the first
N=4 MSA cycle never started. This is the guard working as implemented,
but the failure exposes a validation-semantics bug (see Root cause) —
**the N=16 pilot is NOT cleared.**

## Run identity

- RUN_DIR: `results/runs/base_retention_msa_probe_msa_20260719_161111`
- Launch: `MSA_NUM_SEEDS=4 MSA_EVAL_DEVICES=cuda:4,cuda:5
  bash results/launchers/msa_probe/start_msa.sh --i-know-what-i-am-doing`
- Approved deviation from LAUNCH_CHECKLIST: this machine has 6 GPUs
  (0–5), not 8. Eval devices were reassigned to `cuda:4,cuda:5`
  (user-approved). A stray `bidpilot` process on GPU 0 was killed
  before launch (user-approved). Eval devices were never exercised.

## Pre-launch checks (all passed)

1. `pytest tests/test_msa_probe.py -q`: 110 passed.
2. `continuation_guard --verify-only`: `BOOTSTRAP OK: 6/6 checkpoints
   verified (hash, mapping, fields)`.
3. Disk headroom: 228 GB free.
4. GPUs 0–5 free after killing the stray process.

## Timeline

| Time (UTC+8) | Event |
|---|---|
| 16:11:11 | Arm launched; launcher, guard, sidecar, watchdog all up; six workers staggered at 180 s |
| 16:11–16:47 | Sidecar correctly idle at the double gate ("readiness marker missing"); never touched `MSA_HOLDS.json`, never wrote `MSA_BUSY.json` |
| ~16:48 | Worker 0 produced its first committed tick: prefix `000251`, frame 5,080,000,000 (baseline + 80M) |
| 16:48:11 | Guard first-tick validation failed → wrote `GUARD_FAILED` → stopped the arm; sidecar and watchdog exited cleanly |
| ~16:55 | Five Isaac trainers found still RUNNING after the guard's SIGTERM; manually escalated SIGTERM → SIGKILL; all GPUs freed |

## GUARD_FAILED payload

```
worker 0: first tick tolerance 0.016849003394548208
!= manifest tip tolerance 0.023112487509668322 (atol 1e-06);
tolerance resume failed
```

## Root cause

`--success_tolerance_resume` DID work. `policy_00/train.log` shows:

```
[RESUME] PBT restart from source_policy=None: success_tolerance=0.023112487509668322
```

What actually happened: between resume and the first committed tick
(+80M frames), the env-side mastery/adaptive_fast curriculum performed
three legitimate shrinks:

```
0.023112 -> 0.020801  (nsucc_mean  7.17)
         -> 0.018721  (nsucc_mean 25.36)
         -> 0.016849  (nsucc_mean 21.06)
```

The original 5B run's 0.022–0.028 deadlock depended on accumulated
observer gate state (retention windows, danger zone, stable-tick
counters). A fresh continuation starts with an empty
`curriculum_control/`, so those gates reset and the curriculum resumes
shrinking immediately. The guard's first-tick check assumes the live
tolerance still equals the manifest tip tolerance (atol 1e-6) at the
first committed tick; that assumption conflicts with real continuation
dynamics, so it misclassifies "curriculum legally moved" as "tolerance
resume failed".

## What this run DID verify on real GPUs (first time)

- Full launcher chain: guard bootstrap, staggered 6-worker warm start,
  per-worker `--success_tolerance_resume` propagation, first committed
  tick production.
- Sidecar double-marker gate: idled correctly for ~37 minutes without
  starting a cycle, without touching holds or busy markers.
- Guard fail-closed stop on validation failure.
- Watchdog exits quietly when the arm stops with no holds and no busy
  marker.
- No NCCL timeout, CUDA error, NaN, or segfault anywhere in the run.

Not reached (no data): hold smoke, 27 eval cells, MSA cycle phases,
COMMITTING/acks, +100M cap / `ARM_COMPLETE.json`.

## Final marker state

Present: `GUARD_FAILED`. Absent: `FIRST_TICK_VALIDATED.json`,
`HOLD_SMOKE_*`, `MSA_BUSY.json`, `MSA_HOLDS.json`,
`COMMIT_TIMEOUT.json`, `ARM_COMPLETE.json` — consistent with
fail-closed semantics.

## Incidental scientific observation (informal)

With no MSA intervention at all, worker 0 shrank 0.0231 → 0.0168 within
80M frames at env-window nsucc_mean 21–25. This suggests the original
deadlock is more likely attributable to accumulated gate state than to
policy capability. Caveats: 80M frames only, env-window statistics, no
controlled evaluation. If this holds, a plain BASE_CONTINUATION arm may
already answer the gate-state question, and MSA "due" detection needs
re-examination (workers may never look stuck on fresh continuation).

## Blockers before any re-launch (require code changes)

1. **Guard first-tick tolerance validation semantics.** Must accept
   "resume value correct + subsequent legal curriculum movement" —
   e.g. validate the `[RESUME]` log value, or use a direction/range
   check, or freeze env-side shrink during the readiness window.
2. **`stop_arm`/`stop_worker` SIGTERM is insufficient for Isaac
   trainers.** Five trainers kept training ~7 minutes after SIGTERM
   until manual SIGKILL. Stop paths need SIGTERM → grace → SIGKILL
   escalation.

## N=16 pilot clearance

Not cleared. Fix the two blockers above, then re-run the execution
smoke from the start.
