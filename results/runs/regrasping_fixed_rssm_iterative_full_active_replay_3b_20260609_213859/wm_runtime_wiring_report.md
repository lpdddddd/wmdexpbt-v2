# WM runtime wiring report — `regrasping_fixed_rssm_iterative_full_active_replay_3b_20260609_213859`

- Workspace: `/root/autodl-tmp/wmdexpbt/results/regrasping_fixed_rssm_iterative_full_active_replay_3b_20260609_213859/dexpbt_8gpu_ws`
- `wm_imagined/` dir present: **False**
- `world_model_teachers/` dir present: **False**

## Verdict

WM imagined evaluator: NOT WIRED at runtime. | Iterative teacher: NOT WIRED at runtime. | Active WM ranking: NEVER engaged (top fallback reason: [('feature_off', 285)]).

## Imagined evaluator (per-policy disk mailbox)

_No `wm_imagined/<pidx>.json` files written. The runtime wiring is OFF or never reached the disk mailbox._

## Teachers (per-policy index.json)

_No `world_model_teachers/policy_<pidx>/teacher_index.json` files. TeacherManager NOT wired._

## PBT observer side: WM score usage per policy

| pidx | last_frames | last_tol | last_nsucc | floor_stage | wm_used / n | shadow_changed_max | score_real | score_wm | Δ |
|-----:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 1240M | 0.0259 | 22.3 | none | 0/53 | 0 | 42.32 | 0.00 | -42.32 |
| 1 | 1140M | 0.0362 | 30.6 | none | 0/30 | 0 | 55.66 | 0.00 | -55.66 |
| 2 | 1260M | 0.0264 | 28.4 | none | 0/50 | 0 | 49.87 | 0.00 | -49.87 |
| 3 | 1180M | 0.0245 | 28.2 | none | 0/49 | 0 | 50.88 | 0.00 | -50.88 |
| 4 | 1280M | 0.0233 | 23.8 | none | 0/53 | 0 | 45.52 | 0.00 | -45.52 |
| 5 | 1220M | 0.0310 | 31.4 | none | 0/50 | 0 | 56.66 | 0.00 | -56.66 |

## Fallback reason histogram (per-tick)

- `feature_off`: 285

## Aggregate counters

- Imagined evaluator valid count: **0** of **0** mailbox entries
- Teacher promotions: **0**, rejections: **0**
- Per-policy ``wm_score_used`` ticks (sum across policies): **0**
- Max ``wm_shadow_changed_selection`` across policies: **0**
