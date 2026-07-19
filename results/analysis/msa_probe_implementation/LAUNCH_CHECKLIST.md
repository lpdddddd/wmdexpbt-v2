# base_retention_msa_probe — Launch Checklist (NOT YET LAUNCHED)

Nothing has been started. Every generated launcher refuses to run unless
invoked with `--i-know-what-i-am-doing` after this checklist is signed off.

## Launchers (generated, syntax-checked, never executed)

| arm | script |
|---|---|
| BASE_CONTINUATION | `results/launchers/msa_probe/start_base_continuation.sh` |
| GATE_RELAX | `results/launchers/msa_probe/start_gate_relax.sh` |
| MSA | `results/launchers/msa_probe/start_msa.sh` |
| MSA_PERMUTED_ROUTE | `results/launchers/msa_probe/start_msa_permuted_route.sh` |

Each has a machine-readable `<arm>.config.json` next to it, embedding the
resume checkpoints, seed manifest identity, bootstrap parameters and the
frame-ledger contract.

## Pre-launch checks

1. [ ] All six 5B tip checkpoints still exist and pass zip/torch-load
       integrity (paths in `CONFIG_DIFF.md`; re-run
       `python -m dexpbt_isaaclab.msa.arms --manifest-csv ... --out-dir ...`
       to re-resolve — it fails closed on any missing file).
2. [ ] GPUs 0–5 idle; no other training/eval jobs (`nvidia-smi`).
3. [ ] Disk headroom: each arm writes a new run dir (checkpoints ~180MB x
       6 workers x ~10 ticks for 100M frames ≈ 11GB per arm; MSA arms add
       read-only snapshots + eval raw ≈ +3GB).
4. [ ] `python -m pytest tests/test_msa_probe.py -q` green (40 tests).
5. [ ] Dry-run regenerates cleanly:
       `python -m dexpbt_isaaclab.msa.dry_run --out /tmp/trace.md`.
6. [ ] Decide arm ordering. Recommended: run arms SEQUENTIALLY (each needs
       all 6 GPUs). Suggested order: BASE_CONTINUATION → GATE_RELAX → MSA →
       MSA_PERMUTED_ROUTE.
7. [ ] For MSA arms only: confirm the sidecar eval budget. Formal probe
       evaluation is N=256 seeds per (snapshot, tolerance); with the
       one-Isaac-process-per-seed CCRS protocol this is expensive
       (~256 x ~6 min per eval point on one GPU). If needed, reduce
       `--num-seeds` or reserve a dedicated eval GPU BEFORE launch; do not
       change the manifest mid-arm.
8. [ ] Confirm `curriculum_control/` of the new run dir is empty at start
       (fresh arm; provenance intentionally absent → probe immediately due,
       per spec).

## Launch commands (DO NOT RUN until all boxes are checked)

```bash
bash results/launchers/msa_probe/start_base_continuation.sh --i-know-what-i-am-doing
bash results/launchers/msa_probe/start_gate_relax.sh        --i-know-what-i-am-doing
bash results/launchers/msa_probe/start_msa.sh               --i-know-what-i-am-doing
bash results/launchers/msa_probe/start_msa_permuted_route.sh --i-know-what-i-am-doing
```

## Monitoring once launched

- supervisor log: `<run_dir>/launch_master.log`
- MSA sidecar log: `<run_dir>/msa_sidecar.log`
- probe cycle outcomes: `<run_dir>/dexpbt_8gpu_ws/msa_cycles/*.json`
  (intervention id, L_mid table, routing, panel statistic, rho table,
  promoted / first_blocking_gate)
- per-worker provenance:
  `<run_dir>/dexpbt_8gpu_ws/curriculum_control/msa_provenance_00X.json`

## Abort criteria

- any worker's ledger frame regresses (must be monotonic) → kill arm,
  investigate before resuming;
- sidecar eval fails closed (missing per-seed results) repeatedly → stop the
  sidecar (training may continue; probe resumes later);
- nsucc collapse below the original run's collapse thresholds → the
  original gates (unchanged in every arm) handle it; do not intervene
  manually.
