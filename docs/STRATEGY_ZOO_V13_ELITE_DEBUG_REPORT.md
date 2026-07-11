# Strategy Zoo v1.3 Elite Debug Report

Run: `/root/autodl-tmp/wmdexpbt/results/runs/regrasping_pbt_wm_strategy_zoo_v13_elite_2b_debug_20260617_111118`
Config: `strategy_zoo_v1.3_elite_centered_performance_debug`

## Same-frame nsucc

### @ 1B frames
- **elite_bank_actor_distill** (p4): nsucc=21.67
- **elite_bank_actor_distill_strong** (p5): nsucc=27.23
- **wm_pbt_source_selector** (p3): nsucc=30.92

### @ 1B frames
- **elite_bank_actor_distill** (p4): nsucc=26.10
- **elite_bank_actor_distill_strong** (p5): nsucc=32.37
- **wm_pbt_source_selector** (p3): nsucc=36.55

### @ 2B frames
- **elite_bank_actor_distill** (p4): nsucc=34.10
- **elite_bank_actor_distill_strong** (p5): nsucc=34.81
- **wm_pbt_source_selector** (p3): nsucc=33.98

### @ 0B frames
- **elite_bank_actor_distill** (p4): nsucc=20.91
- **elite_bank_actor_distill_strong** (p5): nsucc=12.25
- **wm_pbt_source_selector** (p3): nsucc=20.29

## Throughput vs p0

- **elite_bank_actor_distill**: nsucc_last=36.2545166015625
- **elite_bank_actor_distill_strong**: nsucc_last=35.1007080078125
- **wm_pbt_source_selector**: nsucc_last=37.336181640625

## Profile copy

attempts=3 accepts=3 reject_reason_counts={'before_after_frames': 3, 'copied': 3, 'retention_low': 1, 'same_profile': 1, 'diversity_min': 1, 'target_protected': 4}
source_to_target={'elite_bank_actor_distill -> elite_bank_actor_distill': 3}

## Population @2B

top3_nsucc_mean=34.299601236979164
- p5 elite_bank_actor_distill_strong: nsucc=34.81 retention=None
- p4 elite_bank_actor_distill: nsucc=34.10 retention=None
- p3 wm_pbt_source_selector: nsucc=33.98 retention=None
