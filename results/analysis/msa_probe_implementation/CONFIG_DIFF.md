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
| `--max-iterations` | 44500 | 45390 | +100M worker-local frames (~890 RL epochs at ~112,360 frames/epoch) |
| run dir | `regrasping_cser_..._strict_5b_*` | `base_retention_msa_probe_<arm>_*` | isolation |

`--recover-checkpoint-per-policy` is a new, inert `launch_pbt` option
(default empty = bit-for-bit legacy behavior; mutually exclusive with the
old single-policy `--recover-checkpoint`).

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
  to the supervisor and drives the probe state machine;
- during probe segments the sidecar writes `forced_success_tolerance`
  (tau_plus, then back to tau) into the existing curriculum control JSON —
  the env already honors this field, so no env/trainer code changes;
- the promotion decision replaces the **cross-tolerance retention read**
  with the **same-tau rho** (`p_post/p_pre` at tau, threshold 0.88 = the
  current applicable base-retention value). All other gate numerics stay.
- `CurriculumControlState` gains three inert fields (`msa_enabled=False`,
  `msa_provenance=None`, `msa_state_machine=None`); no runtime code path in
  the original profile branches on them.

Probe parameters (pre-registered in `msa.config.json`):

| parameter | value |
|---|---|
| tau_plus rule | max(0.010, 0.95 * tau) |
| exposure window | 100,000,000 worker-local frames |
| packet size | 8,000,000 frames (x2: common + routed second half) |
| second-half recipients | top-3 by L_mid (tie: ascending worker id) |
| eval N | 256 ordered seeds (`msa_seeds_310000_256_*`), per-seed results kept |
| bootstrap | seed 20260719, 10000 repeats, panel-median 5th percentile |
| rho threshold | 0.88 (same-tau only; p_pre >= 0.01 required) |

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
