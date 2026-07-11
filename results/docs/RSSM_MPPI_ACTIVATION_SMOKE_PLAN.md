# RSSM-MPPI Activation Smoke Plan

Separate diagnostic run to validate RSSM-MPPI candidate pipeline activation. **Not** part of Strategy Zoo v1.3 performance population.

## Background: v1.2 failure modes

From Strategy Zoo v1.2 2B debug TensorBoard / training-curve analysis:

| Policy | Profile | Symptom |
|--------|---------|---------|
| p0/p1/p2 | light RSSM profiles | `mppi_active = 0`, `rssm_candidate_added = 0` |
| p0/p1/p2 | light RSSM profiles | high `actor_distill_reject_bad_source` |
| p4/p5 | elite bank | `best_source_frac_elite ≈ 0.58`, `bank_elite_bank_add_count ≈ 3.94M` |

**Root-cause hypotheses:**

1. **MPPI blocked by uncertainty gate** — belief uncertainty ~0.93–1.48 vs threshold 0.4; MPPI never activates.
2. **RSSM model not promoted/loaded** — no promoted RSSM checkpoint → no belief update → no MPPI candidates.
3. **Candidate gate weight below retrieval floor** — `eir_candidate_min_weight_for_retrieval=0.05` may block RSSM candidates even when computed.
4. **Actor distill source filter** — `eir_actor_distill_allowed_sources` on light profiles may exclude RSSM when RSSM candidate weight is zero.
5. **Shared launcher WM head flags** — conservative imagined-head thresholds may suppress RSSM planner signal in performance run context.

## Smoke run design

| Parameter | Value |
|-----------|-------|
| Purpose | Activate RSSM-MPPI pipeline in isolation |
| Policies | 1–2 (`max_active=1` or `2`) |
| Target frames | 300M–500M |
| Profile | `light_rssm_mppi_actor_distill_debug` (or dedicated debug variant) |
| Launcher | **New** `start_regrasping_rssm_mppi_activation_smoke.sh` (not yet implemented) |
| Mix with v1.3 | **No** — do not add to performance `PROFILE_LIST` |

Suggested policy layout (when implemented):

```text
p0 = light_rssm_mppi_actor_distill_debug   # primary smoke
p1 = light_conservative_rssm_planner_distill  # optional second arm
```

## Pre-flight checks (before smoke launch)

1. Confirm profile has `eir_use_rssm_mppi=True`, `eir_mppi_mode != off`.
2. Lower MPPI uncertainty threshold temporarily for smoke only (e.g. 0.6–0.8) — document in RUN_CONFIG.
3. Enable RSSM promotion logging; verify `world_model/` checkpoint writes.
4. Set `eir_actor_distill_allowed_sources` to include `rssm` for smoke profile.
5. TensorBoard tags to watch listed below.

## Success criteria

| Metric | Target |
|--------|--------|
| `mppi_active` | > 0 sustained after warm-up |
| `rssm_candidate_added` | > 0 |
| `mppi_belief_valid` | > 0 |
| `mppi_fallback_reason` | distribution logged (not 100% single blocker) |
| RSSM model | promoted/loaded at least once |
| Candidate gate | weight >= `eir_candidate_min_weight_for_retrieval` for RSSM arm |
| `actor_distill_reject_bad_source` | decreasing vs v1.2 baseline |

## Failure actions

| Failure | Action |
|---------|--------|
| `mppi_active=0` entire run | sweep uncertainty threshold + RSSM load path |
| RSSM never promoted | inspect seq_wm train cadence + promotion gates |
| candidates added but distill rejects | relax allowed_sources / gate_mode for smoke only |
| OOM | reduce `num-envs` or disable second policy |

## Telemetry checklist

```text
eir/mppi_active
eir/rssm_candidate_added
eir/mppi_belief_valid
eir/mppi_fallback_reason
eir/mppi_uncertainty
eir/candidate_gate_weight
eir/actor_distill_reject_bad_source
eir/best_source_frac_rssm
world_model/rssm_promoted
world_model/rssm_load_success
```

## Relation to v1.3

v1.3 performance run (`strategy_zoo_v1.3_elite_centered_performance_debug`) **excludes** all RSSM-MPPI profiles. RSSM smoke is a parallel diagnostic track only.
