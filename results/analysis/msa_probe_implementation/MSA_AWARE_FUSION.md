# MSA-aware DexPBT Fusion

## Goal

Unify DexPBT learning with the validated MSA curriculum/retention controller
and existing snippet/WM recovery signals:

```text
DexPBT learner + MSA curriculum/retention controller + snippet/WM recovery
```

MSA is the **sole tolerance committer** under the fusion profile.
Snippet/WM may only provide conservative training signal — never advance.

## Profile

| Profile | Role |
|---------|------|
| `cser_dexpbt_conservative` | **A** — DexPBT + Conservative Retention (legacy; MSA off) |
| `msa_aware_cser_dexpbt_conservative` | **B/C** — DexPBT + integrated MSA (+ snippet/WM as already in conservative) |

Fusion is **default-off**. Old profiles and checkpoints keep prior behavior.

## State machine

Per-worker modes: `BUILD` | `PROBE` | `HOLD`.

- **BUILD** (`τ > 3.0 cm`): normal PPO/PBT; mastery shrink is proposal-only;
  no N=16 paired eval; confirmed tolerance updates the worker anchor.
- **MID** (`2.0 < τ ≤ 3.0 cm`): N=4 / bootstrap prefilter; pass → PROBE.
- **STRICT** (`τ ≤ 2.0 cm`): N=16 paired eval (existing pilot path).

Probe target: `τ⁺ = max(γ·τ, τ_min)` with `γ=0.95`, floor `1.0 cm`.

Advance (pilot-aligned):

```text
L_LCB > 0  AND  ρ_LCB ≥ ρ_min   (default ρ_min = 0.88)
```

Failure → restore anchor tolerance, `HOLD` + cooldown, fail-closed on errors.

## Call chain (main training)

1. `train.py` expands `--wm_strategy_profile msa_aware_cser_dexpbt_conservative`
2. Profile sets `msa_integrated_enabled=True` (+ MSA knobs)
3. `DexPBTPbtAlgoObserver` gets `msa_integrated_enabled` and knobs
4. `after_init` → `MSAFusionHook` migrate/seed worker state under `<run_dir>/msa_worker_state/`
5. Env `apply_env_tolerance_curriculum`: fusion path writes `msa_tolerance_proposal` only
6. Sidecar (debug/standalone) or fusion smoke injects `MSA_FUSION_DECISION.json`
7. Observer `after_steps` → `MSAFusionHook.consume_sidecar_decision_file` →
   ADVANCE/HOLD/ROLLBACK; mirrors into curriculum control; TB `msa/*` scalars
8. `_ternary_pbt` / source eligibility: block PROBE / `MSA_BUSY` exploit
9. PBT restart → `recipient_after_pbt_copy` (evidence reset; no donor MSA pass inherit)

Standalone MSA sidecar remains the production paired-eval + durable commit driver
for pilot arms. Fusion reuses its cycle summary via `fusion_adapter.py`, which
fans out **per-worker commands** under `msa_fusion_protocol/` (exactly-once claim;
no shared `MSA_FUSION_DECISION.json` delete race). Trainers claim only their own
command, apply ADVANCE/ROLLBACK with transactional probe anchors, and ack.
Cycle complete requires all target acks.

Fake injector remains test-only (`MSA_FUSION_ALLOW_FAKE_DECISION=1`). Disposable
integrated smoke may use `--fusion-fixture-once` + `MSA_FUSION_ALLOW_EVAL_FIXTURE=1`
to feed episode-level evidence into the **real** `decide_selective_promotion`
path; formal profiles cannot enable the fixture.

## Launch

Fusion (do not start a formal long run from this doc alone):

```bash
source /root/autodl-tmp/wmdexpbt/runtime_root/scripts/env_dexpbt.sh
# Use launch_8gpu_dexpbt.sh with:
#   --profile-per-policy msa_aware_cser_dexpbt_conservative
# and tip recover args as in results/launchers/msa_probe/
# Plus continuation_guard + hold_watchdog + msa.sidecar (see stateful launchers).
```

Old control (A):

```bash
--profile-per-policy cser_dexpbt_conservative
```

## Tests / smoke

```bash
PY=/root/autodl-tmp/wmdexpbt/runtime_root/envs/isaaclab311/bin/python
cd /root/autodl-tmp/wmdexpbt/runtime_root/src/dexpbt_isaaclab
$PY -m pytest tests/test_msa_fusion.py tests/test_msa_fusion_production.py \
  tests/test_msa_probe.py tests/test_msa_selective_budget.py \
  tests/test_msa_state_bootstrap.py -q

# Control-flow smoke (no GPU; fake injector)
MSA_FUSION_ALLOW_FAKE_DECISION=1 $PY -m dexpbt_isaaclab.msa.fusion_smoke --mode control

# Disposable 6-GPU integrated smoke (production sidecar fixture path)
bash /root/autodl-tmp/wmdexpbt/results/launchers/msa_probe/run_msa_fusion_integrated_smoke.sh --i-know-what-i-am-doing
```

See also: `FUSION_PRODUCTION_CLOSURE.md`.

## A/B/C experiments (future)

- **A**: `cser_dexpbt_conservative` (no MSA integration)
- **B**: `msa_aware_cser_dexpbt_conservative` with WM/distill caps disabled if desired
- **C**: full `msa_aware_cser_dexpbt_conservative` (MSA + existing snippet/WM)

## Differences vs pure in-trainer commit

Durable probe packet + paired eval + tolerance pin remain on the **sidecar** path
(validated by N=4/N=16 pilots). Fusion adds observer hooks, proposal-only shrink,
PBT eligibility, worker_state persistence, and decision application. This avoids
a second evaluator and keeps pilot statistics unchanged.
