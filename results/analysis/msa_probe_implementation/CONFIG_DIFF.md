# base_retention_msa_probe — Config Diff vs Original base-retention

Baseline: `results/launchers/start_regrasping_cser_dexpbt_conservative_retention_strict_5b.sh`
(the launcher of the completed strict-5B run
`regrasping_cser_dexpbt_conservative_retention_strict_5b_20260716_130631`).

All four arms share, verbatim, the baseline's full `--extra` flag block
(extracted programmatically by `msa/arms.py::_base_flags_block`, so drift is
impossible), the same task / 8192 envs / PBT cadence / profile list
(`cser_dexpbt_conservative` x6), and matched `--seed 42`.

## Common changes (all arms, continuation mechanics only)

| item | base 5B run | continuation arms | why |
|---|---|---|---|
| warm start | fresh start | `--recover-checkpoint-per-policy` with the six 5B tip checkpoints | resume the completed population |
| tolerance resume | n/a | `--tolerance-resume-per-policy` with the six manifest live tolerances | a bare `--checkpoint` does NOT restore the curriculum tolerance on a fresh run dir (it would reset to 0.075); per-policy `--success_tolerance_resume` restores it explicitly |
| pre-launch verification | n/a | `continuation_guard --verify-only` (fail-closed) | checkpoint existence + size/head sha256 vs `sha256_head` in the config + worker mapping + manifest fields; any failure aborts before launch |
| runtime guard | n/a | `continuation_guard` loop (per-run `continuation_guard.log`) | first committed tick must match the manifest tolerance (≤1e-6) / lineage / ≥5B frame baseline, else the whole arm is stopped; worker-local **+100M hard frame cap** (+1 tick slack) prevents a silent extra ~5B after an epoch/frame-ledger reset |
| `--max-iterations` | 44500 | 45390 | +100M worker-local frames (~890 RL epochs at ~112,360 frames/epoch); the guard cap is the hard backstop |
| run dir | `regrasping_cser_..._strict_5b_*` | `base_retention_msa_probe_<arm>_*` | isolation |

`--recover-checkpoint-per-policy` and `--tolerance-resume-per-policy` are
inert `launch_pbt` options (default empty = bit-for-bit legacy behavior;
the former is mutually exclusive with the old single-policy
`--recover-checkpoint`, the latter requires the former).

Resume checkpoints (resolved from the CCRS manifest, never guessed):

| worker | checkpoint | frame | live tol (m) |
|---|---|---|---|
| p0 | `dexpbt_8gpu_ws/000/000160.pth` | 5,000,000,000 | 0.02311 |
| p1 | `dexpbt_8gpu_ws/001/000160.pth` | 5,000,000,000 | 0.02028 |
| p2 | `dexpbt_8gpu_ws/002/000161.pth` | 5,080,000,000 | 0.02549 |
| p3 | `dexpbt_8gpu_ws/003/000160.pth` | 5,000,000,000 | 0.02366 |
| p4 | `dexpbt_8gpu_ws/004/000160.pth` | 5,000,000,000 | 0.01927 |
| p5 | `dexpbt_8gpu_ws/005/000160.pth` | 5,000,000,000 | 0.02131 |

## Arm 1 — BASE_CONTINUATION

No flag differences beyond the common continuation mechanics. All gates,
thresholds and safety mechanisms identical to the original run.

## Arm 2 — GATE_RELAX

Exactly ONE numeric change:

```diff
- --late_floor_retention_min_retention 0.88
+ --late_floor_retention_min_retention 0.75
```

0.75 is the pre-existing configured hard minimum
(`--late_floor_retention_hard_min_retention 0.75`), not a new relaxation.
Everything else — including `late_floor_retention_hard_min_retention`,
patience, freeze frames, every other gate — is unchanged (asserted by
`TestOriginalProfileUnchanged.test_base_continuation_launcher_keeps_original_gates`,
which diffs the two generated scripts and requires exactly one differing
line).

**Delivery mechanism (important):** profile expansion
(`apply_wm_strategy_profile`) is *authoritative* for non-legacy profiles —
a profile-controlled flag set on the CLI is silently reverted to the
profile-table value at startup, and `late_floor_retention_min_retention`
IS in the `cser_dexpbt_conservative` table (0.88). A CLI-only override
would therefore be a no-op. The GATE_RELAX arm instead selects a dedicated
profile `cser_dexpbt_conservative_gate_relax` (id 21, registered in
`world_model/wm_strategy_profiles.py`), whose table is byte-identical to
`cser_dexpbt_conservative` except `late_floor_retention_min_retention:
0.75`. The launcher passes it via `--profile-per-policy` for all six
workers; the CLI flag is also set to 0.75 for legibility and now matches
the table. This single-key diff is asserted by
`TestArmConfigs.test_gate_relax_goes_through_dedicated_profile` and
validated fail-closed at launcher-generation time
(`msa.arms._validate_gate_relax_profile`). This also survives PBT
restarts: `profile_controlled_restart_skip_flags` strips
profile-controlled flags from restart argv, and the authoritative table
re-applies 0.75.

## Arm 3 — MSA

Zero numeric gate changes. Additions only:

- the MSA probe sidecar (`python -m dexpbt_isaaclab.msa.sidecar`) runs next
  to the supervisor and drives the probe state machine with phase-barrier
  holds (SIGSTOP/SIGCONT of the worker's training tree during eval phases);
- during probe segments the sidecar pins the tolerance through the
  sidecar-owned `msa_forced_success_tolerance` (worker-local tau_plus,
  then back to the worker's own tau) via the locked
  `update_curriculum_control` API — honored by the env ONLY when
  `msa_enabled`, with priority over (and never touching) the
  observer-owned `forced_success_tolerance`;
- eval subprocesses are pinned to `--eval-device cuda:${MSA_EVAL_GPU:-6}`
  (trainers occupy `DEXPBT_GPUS=0-5`); the flag is required, there is no
  cuda:0 default;
- the promotion decision replaces the **cross-tolerance retention read**
  with the **same-tau rho** (`p_post/p_pre` at the worker's OWN tau,
  threshold 0.88 = the current applicable base-retention value). All other
  gate numerics stay.
- `CurriculumControlState` gains five inert fields (`msa_enabled=False`,
  `msa_forced_success_tolerance=0.0`, `msa_provenance=None`,
  `msa_state_machine=None`, `msa_cycle=None`); with `msa_enabled=False`
  no runtime behavior changes (unit-tested bit-identical shrink path).

Probe parameters (pre-registered in `msa.config.json`):

| parameter | value |
|---|---|
| tau_plus rule | max(0.010, 0.95 * tau[w]) per worker (heterogeneous, criterion-local) |
| exposure window | 100,000,000 worker-local frames |
| packet size | 20,000,000 frames = one committed PBT tick (x2: common + routed second half); provenance records OBSERVED committed frames |
| second-half recipients | top-3 by criterion-local L_mid (tie: ascending worker id) |
| eval N | pilot 16 ordered seeds (`msa_seeds_310000_16_*`), per-seed results kept; formal 256 must be requested explicitly (never a default) |
| bootstrap | seed 20260719, 10000 repeats, panel-median 5th percentile (criterion-local conditional panel) |
| rho threshold | 0.88 (worker-local same-tau only; p_pre >= 0.01 required) |

## Arm 4 — MSA_PERMUTED_ROUTE (negative control)

Identical to MSA except: before second-half routing, the L_mid -> worker
mapping is shuffled with pre-registered seed 777001. Total tau_plus dose,
recipient count and evaluation count are unchanged (unit-tested). q_post is
recorded as unequal-dose follow-up only.

## Frame ledger

Probe packets train through the normal supervisor loop under a forced
tolerance, so every probe frame lands on the worker's monotonic frame
ledger and is charged against the same 100M continuation budget
(`max_iterations` already includes them — no free extra training).
