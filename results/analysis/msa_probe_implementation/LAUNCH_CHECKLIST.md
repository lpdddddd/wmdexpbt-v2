# base_retention_msa_probe — Launch Checklist (NOT YET LAUNCHED)

Nothing has been started. Every generated launcher refuses to run unless
invoked with `--i-know-what-i-am-doing` after this checklist is signed off,
and then STILL aborts unless the fail-closed bootstrap verification
(`continuation_guard --verify-only`) passes.

## Launchers (generated, syntax-checked, never executed)

| arm | script |
|---|---|
| BASE_CONTINUATION | `results/launchers/msa_probe/start_base_continuation.sh` |
| GATE_RELAX | `results/launchers/msa_probe/start_gate_relax.sh` |
| MSA | `results/launchers/msa_probe/start_msa.sh` |
| MSA_PERMUTED_ROUTE | `results/launchers/msa_probe/start_msa_permuted_route.sh` |

Each has a machine-readable `<arm>.config.json` next to it, embedding the
resume checkpoints WITH bootstrap head hashes (`sha256_head`), live
tolerances, lineage ids, seed manifest identity, bootstrap parameters and
the frame-ledger contract. Every launcher:

1. runs `continuation_guard --verify-only` and aborts on any failure
   (checkpoint missing/tampered, mapping error, empty manifest field);
2. passes `--recover-checkpoint-per-policy` AND
   `--tolerance-resume-per-policy` (explicit per-worker tolerance restore);
3. starts the `continuation_guard` loop: first-committed-tick
   tolerance/lineage/frame validation (fail-closed, stops the whole arm)
   plus the worker-local **+100M hard frame cap**.

## Pre-launch checks

1. [ ] All six 5B tip checkpoints still exist and match their bootstrap
       hashes: `python -m dexpbt_isaaclab.msa.continuation_guard
       --config results/launchers/msa_probe/<arm>.config.json
       --verify-only` (already verified once at generation time:
       BOOTSTRAP OK 6/6).
2. [ ] GPUs 0–5 idle for training; GPU 6 reserved for MSA eval
       (`MSA_EVAL_GPU`, default 6); no other jobs (`nvidia-smi`).
3. [ ] Disk headroom: each arm writes a new run dir (checkpoints ~180MB x
       6 workers x ~10 ticks for 100M frames ≈ 11GB per arm; MSA arms add
       read-only snapshots + eval raw ≈ +3GB).
4. [ ] `python -m pytest tests/test_msa_probe.py -q` green (72 tests).
5. [ ] Dry-run regenerates cleanly (heterogeneous taus, 20M packets):
       `python -m dexpbt_isaaclab.msa.dry_run --out /tmp/trace.md`.
6. [ ] Decide arm ordering. Recommended: run arms SEQUENTIALLY (each needs
       all 6 training GPUs). Suggested order: BASE_CONTINUATION →
       GATE_RELAX → MSA → MSA_PERMUTED_ROUTE.
7. [ ] MSA arms: pilot eval is N=16 seeds (launcher default,
       `MSA_NUM_SEEDS`); do NOT raise to the formal N=256 for the first
       pilot. Confirm eval GPU is free for ~1.6h per eval point.
8. [ ] **GPU-unverified item — supervised first hold.** SIGSTOP/SIGCONT
       phase holds have been validated in simulation only. On the FIRST
       MSA cycle, watch `msa_sidecar.log` and `nvidia-smi` through
       PRE_EVAL: confirm the held trainer stops producing ticks, the eval
       runs on the eval GPU, and SIGCONT resumes training cleanly. Abort
       the arm if the trainer crashes on resume.
9. [ ] Confirm `curriculum_control/` of the new run dir is empty at start
       (fresh arm; provenance intentionally absent → probe immediately
       due, per spec).

## Launch commands (DO NOT RUN until all boxes are checked)

```bash
bash results/launchers/msa_probe/start_base_continuation.sh --i-know-what-i-am-doing
bash results/launchers/msa_probe/start_gate_relax.sh        --i-know-what-i-am-doing
bash results/launchers/msa_probe/start_msa.sh               --i-know-what-i-am-doing
bash results/launchers/msa_probe/start_msa_permuted_route.sh --i-know-what-i-am-doing
```

## Monitoring once launched

- supervisor log: `<run_dir>/launch_master.log`
- continuation guard: `<run_dir>/continuation_guard.log`; a fail-closed
  stop leaves `<run_dir>/GUARD_FAILED` or `<run_dir>/CAP_REACHED` with the
  reason
- MSA sidecar log: `<run_dir>/msa_sidecar.log`
- probe cycle outcomes: `<run_dir>/dexpbt_8gpu_ws/msa_cycles/*.json`
  (intervention id, per-worker tau/tau_plus, L_mid table, routing, panel
  statistic, rho table, promoted / first_blocking_gate, resumed flag)
- per-worker MSA state: `msa_provenance` / `msa_state_machine` /
  `msa_cycle` fields inside
  `<run_dir>/dexpbt_8gpu_ws/curriculum_control/00X.json`

## Abort criteria

- `GUARD_FAILED` marker (first-tick tolerance/lineage/frame mismatch) →
  the guard already stopped the arm; investigate the resume wiring before
  any relaunch;
- `CAP_REACHED` marker → the +100M budget is exhausted; this is the normal
  end of a continuation arm;
- any worker's ledger frame regresses (must be monotonic) → kill arm,
  investigate before resuming;
- sidecar eval fails closed (missing per-seed results) repeatedly → stop
  the sidecar (training may continue; the pending cycle resumes
  idempotently later);
- a worker left at tau_plus after a sidecar crash → should not happen
  (cleanup + guard); if observed, restart the sidecar (it resumes and
  reverts) and file a bug;
- nsucc collapse below the original run's collapse thresholds → the
  original gates (unchanged in every arm) handle it; do not intervene
  manually.
