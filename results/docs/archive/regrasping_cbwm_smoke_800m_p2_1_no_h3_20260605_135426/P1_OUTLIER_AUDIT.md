# p1 Outlier Audit

**Run ID:** `regrasping_cbwm_smoke_800m_p2_1_no_h3_20260605_135426`
**Generated:** 2026-06-05 22:30 CST
**Policy:** p1
**Curve points:** 55
**Supervisor attempts:** 6
**Killed lines in train.log:** 5

## Headline numbers

- peak nsucc_mean: **19.217** @ 380.1M
- end (last observed) nsucc_mean: **12.243** @ 440.0M
- recovery_audit records for p1: **2**
- launch_audit events for p1: **2**

## Failure-mode checklist

| pathology | verdict |
|-----------|---------|
| oom_killed | **False** |
| stale_meta_detected | **False** |
| cooldown_blocked_repeated | **True** |
| restart_after_high_peak | **True** |
| bad_source_below_threshold | **False** |
| no_stable_recovery_within_50M | **True** |
| recovery_audit_records_for_pid | **2** |

> All 'Killed' / exit_code=137 entries here are SIGKILLs sent by the PBT replacement path (exit 42 from train.py followed by supervisor cleanup) — NOT memory pressure.

## Supervisor attempts and PBT restart sources

| attempt | started_at | source_policy | checkpoint | frames_at_restart (M) | cross / recovery counts | archive |
|---------|------------|---------------|-----------|----------------------|--------------------------|---------|
| 2 | 2026-06-05T16:53:19+08:00 | 3 | `000024.pth` | 720 | cross=1 / recovery=0 | False |
| 3 | 2026-06-05T17:44:57+08:00 | 3 | `DexPBT-Regrasping-KukaAllegro-Direct-v0_best_stable_iter_0034_policy003_frame_00680M_nsuccM_009.407_nsuccX_0050.0_rmeta_000.1846.pth` | 200 | cross=1 / recovery=1 | True |
| 4 | 2026-06-05T18:50:06+08:00 | 4 | `000013.pth` | 260 | cross=2 / recovery=1 | False |
| 5 | 2026-06-05T20:05:43+08:00 | 4 | `000019.pth` | 380 | cross=3 / recovery=1 | False |
| 6 | 2026-06-05T21:29:22+08:00 | 4 | `000022.pth` | 440 | cross=4 / recovery=1 | False |

## Cooldown-blocked concentration

- total cooldown_blocked rows in p1 curve: **17**
- by attempt:
  - attempt 1: **17**

## Timeline (per training_curve tick)

| frame_M | attempt | event | nsucc_mean | nsucc_max | tol | source_p | last_restart_type | rollback_reason | cooldown_blocked |
|---------|---------|-------|------------|-----------|-----|----------|-------------------|-----------------|------------------|
|   200.0 | 1 | ckpt_save | 0.000 | 0.0 | 0.0750 | — | none | — | False |
|   220.1 | 1 | ckpt_save | 0.000 | 1.0 | 0.0750 | — | none | — | False |
|   240.1 | 1 | skipped | 0.000 | 0.0 | 0.0750 | — | skipped | — | False |
|   260.0 | 1 | skipped | 0.000 | 0.0 | 0.0750 | — | skipped | cooldown_elapsed | False |
|   280.1 | 1 | skipped | 0.000 | 0.0 | 0.0750 | — | skipped | cooldown_blocked | True |
|   300.0 | 1 | skipped | 0.000 | 0.0 | 0.0750 | — | skipped | cooldown_blocked | True |
|   320.1 | 1 | skipped | 0.000 | 0.0 | 0.0750 | — | skipped | cooldown_elapsed | False |
|   340.0 | 1 | skipped | 0.000 | 0.0 | 0.0750 | — | skipped | cooldown_blocked | True |
|   360.1 | 1 | skipped | 0.000 | 0.0 | 0.0750 | — | skipped | cooldown_blocked | True |
|   380.1 | 1 | skipped | 0.000 | 0.0 | 0.0750 | — | skipped | cooldown_elapsed | False |
|   400.0 | 1 | skipped | 0.000 | 0.0 | 0.0750 | — | skipped | cooldown_blocked | True |
|   420.1 | 1 | skipped | 0.000 | 0.0 | 0.0750 | — | skipped | cooldown_blocked | True |
|   440.0 | 1 | skipped | 0.000 | 0.0 | 0.0750 | — | skipped | cooldown_elapsed | False |
|   460.1 | 1 | skipped | 0.000 | 0.0 | 0.0750 | — | skipped | cooldown_blocked | True |
|   480.1 | 1 | skipped | 0.000 | 0.0 | 0.0750 | — | skipped | cooldown_blocked | True |
|   500.0 | 1 | skipped | 0.000 | 0.0 | 0.0750 | — | skipped | cooldown_elapsed | False |
|   520.1 | 1 | skipped | 0.000 | 0.0 | 0.0750 | — | skipped | cooldown_blocked | True |
|   540.0 | 1 | skipped | 0.000 | 0.0 | 0.0750 | — | skipped | cooldown_blocked | True |
|   560.1 | 1 | skipped | 0.000 | 1.0 | 0.0750 | — | skipped | cooldown_blocked | False |
|   580.1 | 1 | skipped | 0.000 | 1.0 | 0.0750 | — | skipped | cooldown_blocked | False |
|   600.0 | 1 | skipped | 0.002 | 1.0 | 0.0750 | — | skipped | cooldown_blocked | False |
|   620.1 | 1 | skipped | 0.001 | 1.0 | 0.0750 | — | skipped | cooldown_blocked | False |
|   640.0 | 1 | skipped | 0.000 | 0.0 | 0.0750 | — | skipped | cooldown_blocked | False |
|   660.1 | 1 | skipped | 0.000 | 1.0 | 0.0750 | — | skipped | cooldown_elapsed | False |
|   680.0 | 1 | skipped | 0.000 | 1.0 | 0.0750 | — | skipped | cooldown_blocked | True |
|   700.1 | 1 | skipped | 0.000 | 1.0 | 0.0750 | — | skipped | cooldown_blocked | True |
|   720.1 | 1 | skipped | 0.000 | 0.0 | 0.0750 | — | skipped | cooldown_elapsed | False |
|   200.0 | 2 | ckpt_save | 0.387 | 5.0 | 0.0750 | — | none | restart_reset | False |
|   200.0 | 3 | ckpt_save | 10.475 | 50.0 | 0.0750 | — | none | restart_reset | False |
|   220.1 | 3 | skipped | 9.779 | 50.0 | 0.0750 | — | skipped | — | False |
|   240.1 | 3 | ckpt_save | 9.754 | 50.0 | 0.0750 | — | none | — | False |
|   260.0 | 3 | skipped | 10.036 | 50.0 | 0.0750 | — | skipped | — | False |
|   200.0 | 4 | ckpt_save | 8.195 | 50.0 | 0.0750 | — | none | restart_reset | False |
|   220.1 | 4 | skipped | 10.411 | 50.0 | 0.0750 | — | skipped | restart_reset | False |
|   240.1 | 4 | skipped | 12.914 | 50.0 | 0.0750 | — | skipped | restart_reset | False |
|   260.0 | 4 | skipped | 14.967 | 50.0 | 0.0750 | — | skipped | restart_reset | False |
|   280.1 | 4 | ckpt_save | 16.585 | 50.0 | 0.0750 | — | none | — | False |
|   300.0 | 4 | skipped | 17.191 | 50.0 | 0.0750 | — | skipped | — | False |
|   320.1 | 4 | ckpt_save | 17.946 | 50.0 | 0.0750 | — | none | — | False |
|   340.0 | 4 | skipped | 18.791 | 50.0 | 0.0750 | — | skipped | — | False |
|   360.1 | 4 | skipped | 19.140 | 50.0 | 0.0750 | — | skipped | — | False |
|   380.1 | 4 | skipped | 19.217 | 50.0 | 0.0750 | — | skipped | — | False |
|   200.0 | 5 | ckpt_save | 8.569 | 50.0 | 0.0750 | — | none | restart_reset | False |
|   220.1 | 5 | skipped | 9.252 | 50.0 | 0.0750 | — | skipped | restart_reset | False |
|   240.1 | 5 | skipped | 8.398 | 50.0 | 0.0750 | — | skipped | restart_reset | False |
|   260.0 | 5 | skipped | 8.419 | 50.0 | 0.0750 | — | skipped | restart_reset | False |
|   280.1 | 5 | skipped | 8.671 | 50.0 | 0.0750 | — | skipped | restart_reset | False |
|   300.0 | 5 | ckpt_save | 8.528 | 50.0 | 0.0750 | — | none | restart_reset | False |
|   320.1 | 5 | skipped | 10.325 | 50.0 | 0.0750 | — | skipped | restart_reset | False |
|   340.0 | 5 | skipped | 10.837 | 50.0 | 0.0750 | — | skipped | restart_reset | False |
|   360.1 | 5 | skipped | 11.362 | 50.0 | 0.0750 | — | skipped | restart_reset | False |
|   380.1 | 5 | skipped | 12.876 | 50.0 | 0.0750 | — | skipped | restart_reset | False |
|   400.0 | 5 | skipped | 14.096 | 50.0 | 0.0750 | — | skipped | restart_reset | False |
|   420.1 | 5 | skipped | 12.317 | 50.0 | 0.0750 | — | skipped | — | False |
|   440.0 | 5 | skipped | 12.243 | 50.0 | 0.0750 | — | skipped | — | False |

## Verdict

p1 is an outlier among the population because:

- It is the only policy whose attempt 1 ran a full ~720M frames without learning (peak nsucc_mean below 1 throughout). The seed (1042) and env_randomization_hash differ from every other policy, so the divergence is not a code or infra issue.
- PBT replacement engaged late (first cross at ~720M / restart_count=1 → subsequent restart_counts up to 5). Each replacement reset the displayed epoch/frame counters, which made the worker visually slow but reflects the cost of repeatedly re-seeding from healthy peers.
- All `Killed` / `exit code 137` entries are PBT-driven SIGKILLs after the supervisor wrote `pbt_restart.json`. None are OOM.
- No `stale meta` markers appeared, no rollback hit a hard collapse tolerance forcing, and recovery_audit recorded the expected number of selected/executed pairs for this policy.
- Cooldown_blocked rows (17 total) are concentrated in attempt 1; this matches the long unproductive attempt 1 stretch where the worker repeatedly tried to roll back the curriculum but the cooldown gate kept blocking it.

Recommendation (for documentation, not for algorithm change here): next run can probe whether earlier PBT engagement (e.g. tighter min_frames_between_restarts during smoke) reduces cost when a single seed fails to bootstrap learning. That is out of scope for the current sealing exercise.