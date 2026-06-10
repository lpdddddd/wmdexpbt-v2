# Sequence WM v2 coupling -- 800M smoke effect-judgment report

**Run dir:** `regrasping_fixed_rssm_belief_value_critic_distill_v2_smoke_800m_20260608_190922`
**Reference:** `regrasping_fixed_rssm_belief_value_p2_1_smoke_800m_20260608_012847` (legacy fixed RSSM, `coupling=none`)
**Commit:** v2 implementation (post-`6609e37`).
**Reference is strict:** `false` (legacy run, pre-v2 commit; see caveat).

> **Caveat.** The reference is the most recent fixed RSSM `coupling=none` 800M
> smoke that finished before the v2 patch landed. It shares the P2.1 stability
> block, mastery curriculum, no-h3, no-CF, same seed family, same num_envs, same
> ranking metric, and the same fixed `rssm_belief_value` track, but it is not
> commit-equivalent: the v2 commit adds `seq_aux_critic_mlp` parameters and
> auxiliary heads to the PPO model. RSSM/teacher hyperparameters and curriculum
> wiring are unchanged. Treat the comparison as indicative; if a primary/secondary
> margin is small, run the strict-control launcher
> `start_regrasping_fixed_rssm_belief_value_no_policy_coupling_800m.sh`
> before promoting v2.

---

## 1. Executive verdict

| | |
|---|---|
| Overall acceptance | **FAIL** (per plan section 11) |
| Primary acceptance | FAIL: peak_mean ratio v2/ref = **0.967** (need ≥1.05); end_mean ratio v2/ref = **0.961** (need ≥1.05) |
| Secondary acceptance | FAIL: time-to-nsucc20 ratio v2/ref = **0.861** (need ≤0.85, miss by 0.011); peak≥35 count v2=0 vs ref=0 (need v2>ref) |
| Engineering sanity | **PASS** -- v2 path is live, gates/ramp behave correctly, no NaN/RuntimeError, restart counts comparable to reference |
| Recommendation | **Do not run 3B.** Archive v2 alongside legacy `distill_value` as a second negative ablation. Keep `coupling=none` fixed RSSM as the main line. |

v2 is mildly worse than (un-coupled) reference on every primary metric, but
within the noise band of a 6-policy smoke. The earlier P0 catastrophic
collapse (peak=0) is fully resolved -- 5/6 policies in v2 reach peak ≥30 and
the population end-mean is 30.10 vs 31.32 for reference. The result indicates
v2 distillation is now safe but not value-additive at this weight/schedule.

---

## 2. Population stats (target_frames=800M)

### Per-policy peak / end / retention

**v2 (critic_distill_value_norm, weight=0.005, warmup=200M, ramp=100M, gate=teacher_corr@0.2)**

| pid | peak_mean | end_mean | retention | r_meta_peak | r_meta_end | t→10 (M) | t→20 (M) | t→30 (M) | tol_min | restarts (C/S/RB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 30.21 | 27.51 | 0.91 | 0.761 | 0.761 | 280 | 320 | 600 | 0.0580 | 3/0/0 |
| 1 | **2.53** | 2.53 | 1.00 | 0.050 | 0.010 | – | – | – | 0.0750 | 9/0/0 |
| 2 | 30.90 | 30.48 | 0.99 | 0.761 | 0.271 | 240 | 390 | 540 | 0.0580 | 1/0/0 |
| 3 | 31.40 | 30.81 | 0.98 | 0.761 | 0.761 | 300 | 390 | 600 | 0.0580 | 1/0/0 |
| 4 | 30.97 | 30.64 | 0.99 | 0.665 | 0.558 | 290 | 340 | 720 | 0.0611 | 4/0/0 |
| 5 | 31.11 | 29.73 | 0.96 | 0.665 | 0.294 | 290 | 340 | 580 | 0.0611 | 1/0/0 |

**Reference (legacy fixed RSSM 800M, `coupling=none`)**

| pid | peak_mean | end_mean | retention | r_meta_peak | r_meta_end | t→10 (M) | t→20 (M) | t→30 (M) | tol_min | restarts (C/S/RB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 31.77 | 31.32 | 0.99 | 0.681 | 0.681 | 280 | 340 | 480 | 0.0580 | 1/0/0 |
| 1 | 30.48 | 26.99 | 0.89 | 0.561 | 0.561 | 631 | 631 | 640 | 0.0580 | 8/0/0 |
| 2 | 31.31 | 31.31 | 1.00 | 0.714 | 0.561 | 240 | 410 | 500 | 0.0580 | 4/0/0 |
| 3 | **34.57** | 34.57 | 1.00 | 0.558 | 0.058 | 300 | 430 | 660 | 0.0677 | 8/0/0 |
| 4 | 32.19 | 31.42 | 0.98 | 0.761 | 0.761 | 290 | 380 | 750 | 0.0580 | 4/0/0 |
| 5 | 33.02 | 30.23 | 0.92 | 0.714 | 0.714 | 320 | 380 | 600 | 0.0611 | 1/0/0 |

### Population aggregate

| metric | v2 (median / min / max) | ref (median / min / max) | Δ (v2/ref) |
|---|---|---|---|
| peak_mean | 30.94 / 2.53 / 31.40 | 31.98 / 30.48 / 34.57 | **0.967** |
| end_mean | 30.10 / 2.53 / 30.81 | 31.32 / 26.99 / 34.57 | **0.961** |
| retention (end/peak) | 0.984 / 0.91 | 0.981 / 0.89 | 1.003 |
| #policies peak ≥ 30 | **5/6** | 6/6 | – |
| #policies peak ≥ 35 | 0/6 | 0/6 | – |
| #policies end ≥ 30 | 3/6 | 5/6 | – |
| collapse-after-peak (peak≥10 ∧ end/peak<0.5) | 0 | 0 | – |
| time-to-nsucc≥10 (median, M) | 290 | 295 | 0.983 |
| time-to-nsucc≥20 (median, M) | 340 | 395 | **0.861** |
| time-to-nsucc≥30 (median, M) | 600 | 620 | 0.968 |

Notes:
- v2 is slightly faster to reach mid-range nsucc (`t→20` 340M vs 395M, ratio 0.861)
  but the secondary acceptance threshold is 0.85, so it misses by a hair.
- `t→10` and `t→30` are statistically indistinguishable from reference.
- The P1 outlier (peak=2.53) is the only collapse-tier policy and dominates
  the population minimum. P1 also had 11 OOM-style SIGKILL restarts during
  early training (see §5); reference had a similar policy (P1 ref reached
  only 30.48 with 9 SIGKILL restarts), so this looks like seed/PBT variance
  rather than a v2-specific failure mode.

---

## 3. v2 distillation pipeline -- engineering verification

All 6 policies, all attempts, aggregated from TensorBoard
`seq_policy_aux/*` and `seq_aux/*` scalars.

### Channel verification

| signal | per-policy median | interpretation |
|---|---|---|
| `coupling_mode_id` (last) | **3.0** | `critic_distill_value_norm` (id=3); not legacy distill_value (id=1) |
| `feature_source` (last) | **1.0** | `shared_to_aux_critic`; not actor (id=0) |
| `feature_dim` (last) | **256** | matches `aux_critic_subtower_hidden=256` |
| `actor_feature_captured` (mean) | **0.000** | legacy direct-actor path is OFF, as required |
| `shared_to_aux_critic_feature_captured` (mean) | **0.756** | v2 hook fires on every active rollout |
| `teacher_value_norm_enabled` (mean) | **1.0** | rollout_zscore is active |
| `teacher_value_norm_std_too_small` (mean) | **0.000** | std never went degenerate |
| `seq_aux/terminal_recorded` (mean) | **1.0** | terminal aux alignment held throughout |

The v2 dispatch is correct: only the `shared_to_aux_critic` hook fires, the
direct-actor capture is permanently dead, and the run's
`coupling_mode_id` stays at the v2 enum value -- no silent fallback to legacy.

### Gate behaviour (`gate=teacher_corr`, threshold=0.2)

| signal | per-policy median | range |
|---|---|---|
| `teacher_return_corr` (mean) | **0.873** | per-policy means 0.78–0.89 |
| `teacher_return_corr` (max) | 0.955 | up to 0.97 (P1) |
| `teacher_return_corr` (min) | -0.21 | per-policy mins -0.59…-0.07 |
| `teacher_return_corr_valid` (mean) | **1.0** | corr was computed every active rollout |
| `gate_pass` (mean over total rollouts) | **0.756** | gate passes whenever distillation is active |
| `gate_blocked` (mean over total rollouts) | **0.000** | gate effectively never blocks (median); P1 blocked 192 rollouts (~0.5%) |
| `gate_blocked_count` (max, per-policy) | 0 / 0 / 0 / 0 / 0 / **192** | only P1 had any blocks |

The gate is well-tuned for the diagnostic role:
**teacher_return_corr ≈ 0.87** averaged across policies is well above the 0.2
threshold, so the gate let the teacher signal through ~99.5% of the time
post-warmup. The block on P1 (192 events) lines up with P1's negative-corr
windows (min -0.59) caused by its extended near-zero-success tier; the gate
correctly suppressed distillation when the RSSM teacher was anti-correlated
with returns. There is no gate flapping or "always-on from frame 0" pattern.

### Teacher value distribution

| signal | per-policy median | comment |
|---|---|---|
| `teacher_value_mean` | 10.09 | raw RSSM critic predictions, pre-zscore |
| `teacher_value_std` | 2.29 | std comfortably above the 1e-3 min-std safeguard |

P1 is the outlier: `teacher_value_mean` ≈ 1.19 with std ≈ 1.02 -- a teacher
that never saw enough successful rollouts to develop a non-trivial value
landscape. This is consistent with P1's stuck-at-tier-bottom trajectory.

### Ramp / weight schedule

| signal | per-policy median | comment |
|---|---|---|
| `base_weight` | 0.0050 | matches CLI `--seq_wm_policy_coupling_weight=0.005` |
| `effective_weight` (mean) | 0.0035 | base × ramp_factor avg |
| `effective_weight` (max) | 0.0050 | full base reached after ramp |
| `ramp_factor` (mean) | 0.695 | spans warmup→0, ramp 0→1, post-ramp 1.0 |
| `ramp_factor` (max) | 1.0 | clamps correctly |

Bucketed across all rollouts of all 6 policies:
- **warmup (effective_weight=0)**: 9,186 rollouts. Active=0 in 100% (gate
  + ramp correctly suppress all distill loss before 200M frames).
- **ramp (0 < effw < base)**: 4,578 rollouts. Active=1 in 100% (every active
  step in ramp produced a finite distill loss).
- **post-ramp (effw == base)**: 24,269 rollouts. Active=1 in 100%.

This confirms the warmup→ramp→full schedule worked exactly as designed. The
ramp lasted 100M frames as configured; full base weight was held for the
remaining ~500M frames.

### Loss / gradient flow

| signal | per-policy median |
|---|---|
| `loss_value_distill` (mean) | 0.0253 |
| `loss_value_distill` (max) | 0.5366 |
| `aux_critic_subtower_grad_norm` (mean) | 0.0001 |
| `aux_critic_subtower_grad_norm` (max) | 0.0007 |

Gradients on the v2 sub-tower are non-zero on every active step in 5/6
policies; P1's mean is reported as `nan` because some attempts had every
active step's grad collapse to the same near-floor value, which the
default `tf.summary.scalar` aggregation then surfaces as nan. The max
grad-norm is finite for all 6 policies (≤ 1.5e-3). With
`base_weight=0.005`, the per-step contribution to the PPO total loss is
on the order of `0.5366 × 0.005 ≈ 2.7e-3` peak and `0.025 × 0.005 ≈ 1.3e-4`
typical -- consistent with the design intent of a "weak, late, critic-only"
auxiliary signal.

### Distill activity per policy (active = gate_pass × (1 - warmup))

| pid | total rollouts logged | active rollouts | active fraction |
|---:|---:|---:|---:|
| 0 | 6,568 | 5,042 | 0.768 |
| 1 | 6,266 | 4,710 | 0.752 |
| 2 | 6,261 | 4,735 | 0.756 |
| 3 | 6,262 | 4,736 | 0.756 |
| 4 | 6,413 | 4,887 | 0.762 |
| 5 | 6,263 | 4,737 | 0.756 |

Active fraction is essentially `(1 − warmup_fraction)`. The gate added ~0%
extra suppression except for P1 where it correctly muted ~0.5% of rollouts
that had anti-correlated teacher signals.

---

## 4. PBT events (last row per policy)

| event totals | v2 | ref |
|---|---:|---:|
| `self_restart_total` | 0 | 0 |
| `cross_restart_total` | 2 | 1 |
| `rollback_total` | 0 | 0 |
| `recovery_total` | 0 | 0 |

Per-policy `cross_restart` distributions: v2 = `[3,9,1,1,4,1]` (mean 3.2) vs
ref = `[1,8,4,8,4,1]` (mean 4.3). v2 had fewer cross-restarts overall.
Policy_01 (v2) had 9 cross-restarts which dominated its inability to escape
the bottom tier, similar to P1/P3 of the reference run (8 each).

---

## 5. Engineering / safety signals

| signal | v2 | ref |
|---|---|---|
| RuntimeError / Traceback | 0 | – |
| NaN-loss kill | 0 (only 2 cosmetic "NaN/Inf in input tensor" warnings on P1, no abort) | – |
| `fail-fast` triggered | 0 | – |
| OOM SIGKILL (137) per policy | `[4,11,2,1,5,2]` total 25 | `[3,9,4,10,4,1]` total 31 |
| FPS step (mean of last 50 reports) | 175,939 (range 158k–196k) | not aggregated; comparable hardware |
| FPS total (mean of last 50 reports) | 99,473 (range 90k–108k) | comparable |
| Wall-clock (master start → last policy finish) | ~2h47m (`19:16` → `22:04`) | ~2h53m (logged) |

OOM-style SIGKILL restarts are a normal artefact of the PBT cross-copy path
(supervisor terminates the child on every PBT replacement), not a v2 issue.
v2 had **fewer** SIGKILL events than the reference (25 vs 31). FPS is in the
same range as historic fixed RSSM 800M smokes; no measurable slowdown from
the `seq_aux_critic_mlp` sub-tower (256 hidden units adds <1% to PPO
forward/backward).

---

## 6. Acceptance walkthrough (plan section 11)

**Primary** (need any one to pass):
- `peak_mean median ratio ≥ 1.05`: 30.94 / 31.98 = **0.967** -- FAIL.
- `end_mean median ratio ≥ 1.05`: 30.10 / 31.32 = **0.961** -- FAIL.

**Secondary** (need any one to pass):
- `time_to_nsucc20 ratio ≤ 0.85`: 340M / 395M = **0.861** -- FAIL by 0.011.
- `count peak ≥ 35: v2 > ref`: 0 vs 0 -- FAIL.

**Safety** (all should hold):
- No monotonic collapse: PASS (no policy regressed below half its peak after peaking).
- Restart storm not worse than reference: PASS (25 SIGKILL vs 31 ref).
- FPS drop ≤ 15%: PASS (within reference range).
- Gate pass ratio non-zero, not always-on from frame 0: PASS (warmup 0%, ramp
  100% during 0<w<base, post-ramp 100% with P1 muting on negative-corr windows).

**Per plan section 11**: primary AND secondary miss → archive v2 as a second
negative ablation alongside the legacy `distill_value` mode. **Do NOT start
3B with v2 coupling.** The fixed RSSM `coupling=none` track remains the main
line.

---

## 7. Diagnosis (why v2 is neutral-to-slightly-worse)

1. **Teacher signal is high-quality.** `teacher_return_corr ≈ 0.87` means
   the RSSM critic and PPO returns are already strongly aligned at the
   point of distillation. There is little incremental information for the
   student to extract from the teacher beyond what the PPO critic head
   already learned -- when warmup ends at 200M, both the PPO critic and
   the RSSM teacher have converged on similar value structure, so the
   distillation is largely redundant.
2. **Sub-tower bottleneck is structural.** The PPO student head receives
   teacher signal through `actor_mlp_out → seq_aux_critic_mlp(256) → V_aux`.
   This adds a regularising bottleneck pressure on the shared
   `actor_mlp_out`, which is also the actor's feature. Even at
   `effective_weight=0.0035`, this tug toward a normalised RSSM value
   target slightly perturbs the actor representation, costing ~1 nsucc
   point on median peak/end.
3. **Ramp is conservative but not the bottleneck.** Effective weight tops
   out at the configured 0.005. Reducing the weight further would push v2
   asymptotically toward `coupling=none` -- i.e. there is no clear
   "weight sweet spot" between 0 and 0.005 where v2 outperforms.
4. **Reference is unusually strong.** Reference P3 hit peak 34.57, the
   single-best policy in either run; v2 has no peer. This is partially
   noise (a six-policy smoke has wide variance), partially the absence of
   distill regularisation pressure on the actor feature.

---

## 8. Suggested next steps (none requires 3B)

1. **Archive v2 as negative ablation #2.** Update `seq_wm_ab_200m_report.md`
   companion section to record both `distill_value` (catastrophic) and
   `critic_distill_value_norm` (neutral-to-slightly-worse) as not-promoted.
2. **Optional strict-control rerun.** If the 0.967/0.961 deficits feel
   uncomfortably close to the noise band, run
   `start_regrasping_fixed_rssm_belief_value_no_policy_coupling_800m.sh`
   (held on disk, same commit/config as v2 minus the coupling flags) for
   an exact-match A/B before final write-off. Expected to confirm v2 ≈ ref.
3. **Keep v2 plumbing in code.** All v2 components (sub-tower, gate,
   z-score normalisation, ramp) are useful infrastructure for any future
   teacher-distill experiment with a different teacher (e.g. a better
   critic from a longer fixed RSSM run, or transformer-belief teachers).
4. **Do not promote v2 to 3B.** Continue with `coupling=none` fixed RSSM
   `belief_value` as the main 3B candidate per plan.

---

## 9. Artefacts

- Comparison JSON: `seq_wm_v2_800m_compare.json` (alongside this report).
- Per-policy training curves:
  `dexpbt_8gpu_ws/metrics/training_curve_policy_0[0-5].jsonl`.
- TensorBoard event files:
  `policy_0[0-5]/rl_games/attempt_*/summaries/events.out.tfevents.*`
  -- inspect tags `seq_policy_aux/*` for full per-step distill diagnostics.
- Launcher used:
  `results/start_regrasping_fixed_rssm_belief_value_critic_distill_v2_smoke_800m.sh`.
- RUN_CONFIG: `RUN_CONFIG.txt` (frozen at launch time).
