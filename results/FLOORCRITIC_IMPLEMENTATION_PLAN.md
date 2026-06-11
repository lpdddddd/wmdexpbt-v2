# FloorCritic Implementation Plan & As-Built (Real-target, two-phase)

WM-FloorCritic / FloorCritic-DexPBT. Task-progress auxiliary critic learning
on REAL rollout transitions. NOT RSSM value distillation, NOT actor imitation,
NOT Dreamer-style latent policy training. Reward / env / PPO actor-critic
structure unchanged. Default-off.

## 0. Two-phase design

- **Phase 0 — detached smoke**: `feature_source=actor_mlp_out_detached`,
  `actor_grad_blocked=true`. The captured 256-d `actor_mlp_out` is detached
  before the aux tower, so the FloorCritic loss updates ONLY the aux tower +
  heads. Engineering validation only; no performance expected.
- **Phase 1 — effective**: `feature_source=actor_mlp_out_shared`. Feature is
  fed WITHOUT detach, so the loss regularises the shared actor/value
  representation (actor_mlp + LSTM). FloorCritic heads NEVER connect the `mu`
  (actor logits) head, so the actor logit mapping is not directly updated.
  Risk bounded by small weight + grad-ratio auto-scale.
- `critic_feature`: enum kept for a future critic-only encoder; under the
  stock rl_games `separate=False` network there is no populated critic
  feature, so it degrades to shared (with a warning).

## 1. Code review results

- Network: stock rl_games `A2CBuilder.Network` (`continuous_a2c_logstd`),
  `agents/rl_games_ppo_cfg.yaml` `network.separate: False`. Shared LSTM +
  `actor_mlp(768->512->256)`; `mu(256->23)` and `value(256->1)` both read the
  same 256-d feature. `critic_mlp` is empty/unused. No critic-only feature.
- Hooks: `seq_policy_aux.register_actor_mlp_capture` captures `actor_mlp_out`,
  but only when `seq_wm_policy_coupling != none`. With the default `none`,
  nothing is hooked. -> FloorCritic installs its OWN independent hook +
  `get_aux_loss` patch.
- `get_aux_loss`: rl_games `calc_gradients` calls `model.get_aux_loss()` after
  the forward and adds returned tensors to the PPO loss before `backward()`.
  FloorCritic chains its patch on top of whatever is installed (base or
  seq-distill).
- Aux fields (`wm_aux_buffer.py`): all 8 are in `EXTENDED_SEQ_AUX_FIELDS`, but
  registered only when `seq_wm_use_task_aux_inputs` OR
  `seq_wm_enable_iterative_teacher` (now also when `wm_floorcritic_enable`).
  `aux_dropped` is registered but NEVER populated by the env, so `drop_risk`
  derives "dropped" from `lifted` 1->0 instead.
- Minibatch alignment: targets built on `(T,N)`, flattened with
  `flat[n*T+t]=x[t,n]` (rl_games `swap_and_flatten01`), sliced by
  `agent.dataset.last_range`.
- lambda(tolerance): reuses the same thresholds as the WM `floor_aware_weighting`
  (`0.02 / 0.012`) and the PBT `floor_stage_classify`; FloorCritic adds a
  `start_tol` (default 0.04) above which lambda=0.

## 2. actor_grad_blocked

- Phase 0 detached: `feat.detach()` -> FC loss reaches neither backbone nor
  actor. Tested: mu/actor_mlp/value grad-from-FC == 0.
- Phase 1 shared: FC never connects `mu`, so `actor_head_grad_from_fc == 0`
  always; `shared_backbone_grad_from_fc > 0` (intended). Tested.
- Hard guarantees in `compute_grad_audit` (tests run a real FC-only backward
  and inspect `.grad`). Runtime logs proxies + the structural `actor_grad_blocked`.

## 3. Files

New:
- `runtime_root/src/dexpbt_isaaclab/dexpbt_isaaclab/world_model/floorcritic.py`
  (`FloorCriticConfig`, `FloorCriticTargetBuilder`/`FloorCriticTargetCache`,
  `FloorCriticHeads` + `attach_floorcritic_heads`, `FloorCriticBridge`,
  `register_floorcritic_capture`, `register_actor_mlp_grad_probe`,
  `compute_floorcritic_loss`, `compute_grad_audit`)
- `runtime_root/src/dexpbt_isaaclab/tests/test_floorcritic.py` (25 tests)
- `results/start_regrasping_floorcritic_smoke_200m.sh` (Phase 0 detached)
- `results/start_regrasping_floorcritic_800m.sh` (Phase 1 shared)
- `results/start_regrasping_floorcritic_1500m.sh` (Phase 1 shared, wrapper)
- `results/analyze_floorcritic_run.py`
- `results/FLOORCRITIC_IMPLEMENTATION_PLAN.md` (this file)

Modified:
- `scripts/train.py`: flags, `FloorCriticConfig` build + conflict warnings,
  EXTENDED aux registration gate now includes `wm_floorcritic_enable`,
  independent hook + grad probe + `_patch_model_get_aux_loss_for_floorcritic`,
  per-rollout target cache build in `patched_play_steps_rnn`, TB logging in
  `patched_train_epoch`. Bridge/cfg attached on `seq_wm_trainer`.
- `scripts/launch_pbt.py`: FloorCritic boolean flags added to `boolean_flags`.

## 4. Flags (BooleanOptionalAction for bools, matching repo convention)

Bools use `--flag` / `--no-flag` (NOT `true|false`), consistent with the rest
of the codebase and `launch_pbt` passthrough.

```
--wm_floorcritic_enable / --no-...            (default off)
--wm_floorcritic_feature_source               actor_mlp_out_detached|actor_mlp_out_shared|critic_feature
--wm_floorcritic_max_grad_ratio               0.1
--wm_floorcritic_auto_scale_by_grad_ratio / --no-...   (default on)
--wm_floorcritic_start_tol                    0.04
--wm_floorcritic_floor_tol                    0.02
--wm_floorcritic_hard_floor_tol               0.012
--wm_floorcritic_horizons                     4,8
--wm_floorcritic_weight                       0.005
--wm_floorcritic_floor_weight                 0.01
--wm_floorcritic_hard_floor_weight            0.02
--wm_floorcritic_near_success_mult            1.5
--wm_floorcritic_dhat_improve_margin          0.005
--wm_floorcritic_drop_dhat_margin             0.01
--wm_floorcritic_actor_detach / --no-...      (compat; feature_source authoritative)
--wm_floorcritic_update_actor / --no-...      (default off)
--wm_floorcritic_use_{success,near_success,dhat_improve,drop_risk} / --no-...
```

## 5. Target construction (real-only, (T,N), per horizon h in {4,8})

valid iff no done in `t..t+h-1` and `t+h < T`. For valid t:
- `near_success_h`: exists k in [1,h]: `d_hat[t+k] <= near_success_mult * tol[t]`
- `success_h`: exists k in [1,h]: `success_cnt[t+k] > success_cnt[t]`
- `dhat_improve_h`: `min_{1..h} d_hat[t+k] <= d_hat[t] - dhat_improve_margin`
- `drop_risk_h`: any of failure==1 / `d_hat` rises by `drop_dhat_margin` /
  `lifted` 1->0 within next h
- pos_rate, valid_fraction, degeneracy recorded per head.

## 6. Loss

```
L_total = L_PPO + lambda_fc(tol) * auto_scale_factor * sum_head BCE(pred, target)[masked]
```
- lambda_fc: 0 above start_tol; weight / floor_weight / hard_floor_weight by
  tolerance tier (masked-mean of `aux_success_tolerance` over the rollout).
- auto_scale (shared only): `fc_feat_grad_norm` (probe on detached clone) over
  an EMA of the PPO backbone grad norm (backward hook on actor_mlp). If ratio
  > max_grad_ratio, scale down to cap backbone perturbation. Detached: ratio 0,
  factor 1.

## 7. Logging (TB, `wm_floorcritic/*`)

active, lambda, loss_total, loss_<head>, pos_rate_<head>, target_valid_fraction,
feature_source_id, actor_head_grad_from_fc, value_head_grad_from_fc,
shared_backbone_grad_from_fc, fc_grad_ratio_to_ppo_grad, auto_scale_factor,
actor_grad_blocked, critic_grad_norm, aux_head_grad_norm.

## 8. Tests — `tests/test_floorcritic.py` (25, all passing)

default-off; target builder (each type + done-boundary + horizon-overflow +
pos_rate/valid_fraction); gradient (detached blocks backbone+actor; shared
updates backbone but not mu); grad-ratio auto-scale (caps; detached=1.0);
lambda tiers + boundaries; feature_source switching + critic degradation +
invalid raises; attach param group; loss grad-bearing + logging; inactive when
tol > start_tol; CLI flag presence + launch_pbt booleans.

Run: `envs/isaaclab311/bin/python -m unittest tests.test_floorcritic`.
Existing `tests.test_floor_recovery_stack` still 57/57 (no regression).

## 9. Launchers (prepared, NOT started)

- `start_regrasping_floorcritic_smoke_200m.sh`: detached, weight 0.005,
  `pbt_wm_score_mode off`, no teacher/imagined/distill/h3/CF. Smoke raises
  `start_tol=0.1` so FloorCritic is active across the 200M (otherwise tol stays
  ~0.075 and lambda=0 the whole run).
- `start_regrasping_floorcritic_800m.sh`: shared, `weight=0.001`
  (floor 0.002 / hard 0.004), production tol thresholds, `max_grad_ratio=0.1`,
  auto-scale on, DexPBT adaptive_fast + source_safe.
- `start_regrasping_floorcritic_1500m.sh`: 800M config, `MAX_ITER=11444`.
- 3B not prepared until 800M/1.5B look healthy.

## 10. Analyzer — `analyze_floorcritic_run.py`

`python analyze_floorcritic_run.py <run_dir> [--baseline <dir>]` ->
`FLOORCRITIC_RUN_REPORT.md/.json`: nsucc peak/end, tolerance trajectory, tol
threshold crossing frames, FC active frames, loss curves, target pos_rate +
degeneracy, actor_grad_blocked always-true check, grad-ratio / auto-scale /
backbone-grad stats, baseline comparison, proceed recommendation. Reads
FloorCritic scalars from rl_games TB event files under
`policy_XX/rl_games/attempt_*/summaries/`.

## 11. Main risks

- Phase 1 shared perturbs the backbone (not actor head): mitigated by small
  weight (0.001 start) + grad-ratio auto-scale + early-nsucc watch; fall back
  to detached or lower weight on regression.
- `aux_dropped` unpopulated -> drop_risk uses failure + d_hat rise + lifted
  1->0.
- `critic_feature` == shared today (separate=False); warned.
- Coexistence with seq distill: independent capture container + chained
  get_aux_loss patch.
- Smoke at 200M would otherwise never activate FC (tol > 0.04 throughout) ->
  smoke raises start_tol.

## 12. Recommendation

Run 200M detached smoke first (engineering + actor_grad_blocked + non-degenerate
targets), then 800M shared from weight=0.001 vs baseline (watch grad ratio +
early nsucc), then 1.5B; 3B only after.
