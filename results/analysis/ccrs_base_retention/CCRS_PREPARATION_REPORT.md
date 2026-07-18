# CCRS 准备报告（base-retention 5B）

日期：2026-07-18　　分支：`base-retention`　　HEAD：`5cb2c7cdafb12e50b9b7db88fdc47e7ef1056f86`
目标 run：`results/runs/regrasping_cser_dexpbt_conservative_retention_strict_5b_20260716_130631`

状态：**准备完成，未执行任何 CCRS 评测，未占用 GPU**。等待确认后启动。

---

## 1. 找到了哪些 checkpoint

Workspace `dexpbt_8gpu_ws/{000..005}/` 保存完整：每个 worker 约 100–103 个 PBT tick
checkpoint（前期每 40M、后期每 20M 一存），每个 .pth 约 180MB，配套 meta json 含
frame / nsucc_mean / curriculum.success_tolerance / lineage_id。总计约 216GB。

12 个关键 checkpoint 全部命中（frame 偏差为 0），并通过 zip 结构 + torch.load(CPU) 校验：

| worker | iter | frame | live tol (m) | live nsucc | 角色 |
|---|---|---|---|---|---|
| p1 | 149 | 4.12B | 0.0237 | 43.4 | peak（目标 4.12B，精确命中） |
| p1 | 158 | 4.84B | 0.0203 | 39.0 | tip（见第 2 节偏差说明） |
| p4 | 141 | 3.48B | 0.0214 | 40.2 | peak（精确命中） |
| p4 | 160 | 5.00B | 0.0193 | 31.8 | tip |
| p5 | 117 | 2.52B | 0.0236 | 45.0 | peak（精确命中） |
| p5 | 160 | 5.00B | 0.0213 | 35.7 | tip |
| p3 | 159 | 4.92B | 0.0237 | 41.2 | peak/target（精确命中） |
| p3 | 160 | 5.00B | 0.0237 | 39.7 | tip |
| p2 | 96 | 1.92B | 0.0365 | 44.8 | peak（精确命中） |
| p2 | 109 | 2.20B | 0.0297 | 44.7 | target |
| p2 | 161 | 5.08B | 0.0255 | 30.6 | tip |
| p0 | 160 | 5.00B | 0.0231 | 33.2 | tip/peak |

tolerance 收紧成对 checkpoint（post-tol ∈ [0.018, 0.0285]）共检出 **37 次 transition**
（p0:8、p1:7、p2:3、p3:4、p4:9、p5:6），全部前/后 checkpoint 都存在，已写入
`tolerance_transitions.csv`；对应 checkpoint 共 83 行进入 manifest（0 缺失，0 损坏）。

产物：
- `checkpoint_manifest.csv`（83 行，含 run_id / git_commit / worker_id / frame /
  checkpoint_path / live_tolerance / live_nsucc / historical_peak_nsucc / lineage_id /
  policy_arch / roles / frame_deviation / zip_ok / partial_sha256 / torch_load_ok）
- `CHECKPOINT_AUDIT.md`、`tolerance_transitions.csv`

## 2. 缺少哪些关键 checkpoint

- **p1 没有 5.00B tick**：p1 训练进度落后，审计时最新 tick 是 iter 158（4.84B，
  tol 0.0203）。p1 和 p2 的训练进程仍在收尾（其余 4 个 worker 已到 5B+）。两个选项：
  等 p1 到 5.00B 后重跑一次 manifest（一条命令），或直接接受 4.84B 作为 p1 tip
  （偏差 160M frames，已在 manifest 中记录）。
- 其余目标 checkpoint 无缺失，无需用邻近 tick 替代。

## 3. 复用了哪些现有 evaluation 代码

- `scripts/eval_fair.py`：直接 import 其 `_load_env_cfg` / `_load_agent_cfg` /
  `_filtered_restore`（含 aux_value_head 过滤），rollout 循环与 tolerance 冻结方式
  （`success_tolerance` / `initial_tolerance` / `target_tolerance` 同置 +
  `last_curriculum_update = 1e9`）与 eval_fair 逐行一致，不存在第二套环境逻辑。
- 沿用既有经验：**一个进程只跑一个评测单元**（Isaac 在 env.close() 后重建会挂死，
  2026-07-15 eval 已验证），由调度器起子进程。
- 部署口径确认：训练 launcher 中 `--no-eir_residual_apply_to_rollout
  --no-eir_use_rssm_mppi --eir_mppi_mode=off`，即 rollout 本来就不使用 residual/MPPI；
  snippet retrieval / WM rerank / actor distillation 只影响训练更新，不影响 rl_games
  player 的纯 actor 前向。因此 policy-only 主评测（纯 player）与正式部署路径一致。

## 4. 新增/修改了哪些文件

新增（无既有文件被修改）：

| 文件 | 作用 |
|---|---|
| `dexpbt_isaaclab/ccrs/__init__.py` | 包说明（单位约定：一律米） |
| `dexpbt_isaaclab/ccrs/manifest.py` | 只读 manifest 构建器 + transition 检测 + 完整性校验（CLI） |
| `dexpbt_isaaclab/ccrs/eval_protocol.py` | 评测单元定义、config hash、resume/skip、指标聚合（Wilson CI、分位数）、单位防呆（cm 当米传会报错） |
| `dexpbt_isaaclab/ccrs/analysis.py` | R(k,τ) 表、D/F/C 分解（恒等式 D=F+C 按构造成立）、配对 seed bootstrap（非配对数据直接拒绝）、预注册判定 A/B/C/D |
| `dexpbt_isaaclab/ccrs/report.py` | capability 曲线 / CCRS heatmap / p4−p1 差值 / p5 peak−tip / D/F/C 图 / precision-retention frontier / CCRS_REPORT.md / ccrs_summary.json；空数据时降级为 dry 模式，不伪造结果 |
| `scripts/ccrs_eval.py` | GPU 单元评测（1 checkpoint × 1 tolerance × 1 seed / 进程）；启动 Isaac 前完成 resume 检查；冻结 policy（player-only、requires_grad=False、断言无 optimizer）；tolerance 漂移则整单报错 |
| `scripts/ccrs_sweep.py` | 无 Isaac 的调度器：读 manifest 生成单元、按 shard 切分、跳过已完成、子进程串行执行、超时保护、聚合出 `ccrs_results.jsonl/.csv`；默认 plan-only，必须显式 `--execute` |
| `tests/test_ccrs_toolkit.py` | 29 个单元测试 |

输出目录（已建）：`results/analysis/ccrs_base_retention/{raw/, figures/}` +
manifest / audit / transitions + dry 模式的 `CCRS_REPORT.md`、`ccrs_summary.json`。

每个原始结果 JSON 记录：git SHA、config hash、checkpoint hash（size+前 4MB sha256）与
路径、reset seed、fixed tolerance（米 + 换算 cm）、evaluation_mode，以及
num_episodes / num_success / success_rate（+Wilson 95% CI）/ nsucc_mean（+95% CI）/
nsucc_std / final_error mean·median·q10·q25·q50·q75·q90，并**保留 per-episode 的
nsucc 与 final_error 数组**（jsonl 中保留，csv 只存聚合），可任意重新聚合。

## 5. 单元测试结果

`python3 -m pytest tests/test_ccrs_toolkit.py -q` → **29 passed**（0.84s，无 GPU）。
覆盖用户要求的 9 项：① tolerance 漂移报错；② evaluator 源码级断言无
optimizer/PBT/backward 路径；③ 同 seed 跨 checkpoint/tolerance 的 reset 恒等（seed 完全
决定 env_cfg.seed）；④ manifest 正确识别 worker/frame/tolerance/transition，缺失
checkpoint 显式标记不伪造；⑤ resume/skip（含损坏结果文件不算完成）；⑥ 合成数据
D=F+C（残差 <1e-12）；⑦ 缺 cell / 非配对 seed / 单 seed 明确报错；⑧ policy-only：
system_on 拒绝启动，eval 源码不引用 snippet/WM/MPPI/residual/distill；⑨ 单位一律米
（传 2.0 判定为 cm 误用直接报错），报告同时显示 cm。

## 6. 完整 sweep 的组合数量

- tolerance 网格：`[0.028, 0.025, 0.023, 0.022, 0.021, 0.020, 0.019, 0.018]`（8 档）
- 默认 seeds：4（10001–10004），可配置
- **主 sweep（推荐先跑）**：12 个关键 checkpoint × 8 tol × 4 seed = **384 单元**。
  足以回答问题 1（p4 vs p1 高精度）、2（crossing）、3（p5 peak vs tip）。
- **可选补充（D/F/C 全量分解）**：p1/p4/p5 的 transition 成对 checkpoint 44 个，
  每个只需评 (a,b) 两档 tol × 4 seed ≈ **352 单元**；全 6 worker 全 transition 全网格
  则是 71×8×4 ≈ 2272 单元（不建议一次性跑）。

## 7. GPU / 耗时 / 成本估计

依据 2026-07-15 同规格评测实测（512 envs × 512 episodes，一进程一评测，含 Isaac
启动）：**每单元约 1–2 分钟，按 2 分钟保守估计**。

| 方案 | 单元数 | 1×GPU | 6×GPU（分 shard） |
|---|---|---|---|
| 主 sweep | 384 | ≈ 12.8 h | ≈ 2.2 h |
| + p1/p4/p5 transition D/F/C | +352 | +11.7 h | +2 h |

显存：单元评测约 10–12GB（V100 32GB 足够）。当前 GPU 全部空闲（p1/p2 训练进程尚在
收尾，只占 GPU1 约 11.7GB 显存、0% 利用率）；若等训练完全结束再跑更干净。
断点续跑：任意中断后重新执行同一命令自动跳过已完成单元。

## 8. 推荐的正式启动命令

主 sweep，6 卡各一个 shard（**未经你确认我不会执行**）：

```bash
cd /root/autodl-tmp/wmdexpbt/runtime_root/src/dexpbt_isaaclab
for i in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$i nohup python3 -m dexpbt_isaaclab.scripts.ccrs_sweep \
    --manifest /root/autodl-tmp/wmdexpbt/results/analysis/ccrs_base_retention/checkpoint_manifest.csv \
    --roles tip,peak,target \
    --out-dir /root/autodl-tmp/wmdexpbt/results/analysis/ccrs_base_retention \
    --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \
    --num-envs 512 --num-episodes 512 --seeds 10001,10002,10003,10004 \
    --execute --shard-index $i --num-shards 6 \
    > /root/autodl-tmp/wmdexpbt/results/analysis/ccrs_base_retention/sweep_shard$i.log 2>&1 &
done
```

跑完后聚合 + 出报告：

```bash
python3 -m dexpbt_isaaclab.scripts.ccrs_sweep --aggregate-only \
  --manifest .../checkpoint_manifest.csv --out-dir .../ccrs_base_retention \
  --task DexPBT-Regrasping-KukaAllegro-Direct-v0
python3 -m dexpbt_isaaclab.ccrs.report \
  --results .../ccrs_results.jsonl --manifest .../checkpoint_manifest.csv \
  --transitions .../tolerance_transitions.csv --out-dir .../ccrs_base_retention
```

预演（不占 GPU，随时可跑）：把 `--execute` 换成 `--plan`。

## 9. 当前仍存在的风险

1. **p1/p2 训练未完全收尾**：建议等训练进程退出后再启动 sweep（避免 GPU/IO 干扰，
   也可顺带拿到 p1 的真 5.00B tip，重跑一次 manifest 即可）。
2. **evaluator 未做过 GPU smoke test**（按你的要求未占卡）。协议层已全部单测，但
   Isaac 启动路径只与 eval_fair 逐行同构、未实测。建议正式 sweep 前先跑 1 个单元
   验证（约 2 分钟、1 张卡），命令见下第 10 节。
3. **seed 与 reset 的对应关系**：seed 通过 `env_cfg.seed` 在建 env 时注入，与
   eval_fair 相同；同 seed 同 num_envs 下初始 reset 一致，但这依赖 Isaac 的确定性
   实现，smoke test 时会顺带核对（同 seed 两次跑同 checkpoint 应产出几乎相同指标）。
4. **每 seed episodes=512 × 4 seeds** 的方差对 |Δ|≥0.05 的判定检验力足够，但如果
   结果落在边界附近，可能需要追加 seeds（工具支持增量追加，不会重算旧单元）。
5. p2 peak（1.92B，tol 0.0365）处于宽松 tolerance 区，其在 0.018–0.028 网格上的
   成功率可能整体偏低，属预期现象，不影响 p1/p4/p5 的核心问题。

## 10. 是否满足正式启动条件

**代码与数据侧：满足。** checkpoint 全部命中且可加载、评测/分析/报告工具齐备、
29 项单测全绿、resume/分片/单位防呆就位。

**唯一未完成项**：1 个单元的 GPU smoke test（你要求先确认）。建议的 smoke 命令
（单卡、约 2 分钟，跑 p4 peak @ 0.020m @ seed 10001）：

```bash
cd /root/autodl-tmp/wmdexpbt/runtime_root/src/dexpbt_isaaclab
CUDA_VISIBLE_DEVICES=2 python3 -m dexpbt_isaaclab.scripts.ccrs_eval \
  --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \
  --run-id regrasping_cser_dexpbt_conservative_retention_strict_5b_20260716_130631 \
  --worker-id 4 --checkpoint-iteration 141 --checkpoint-frame 3480000000 \
  --checkpoint-path /root/autodl-tmp/wmdexpbt/results/runs/regrasping_cser_dexpbt_conservative_retention_strict_5b_20260716_130631/dexpbt_8gpu_ws/004/000141.pth \
  --fixed-tolerance 0.020 --reset-seed 10001 \
  --num-envs 512 --num-episodes 512 \
  --raw-dir /root/autodl-tmp/wmdexpbt/results/analysis/ccrs_base_retention/raw \
  --headless
```

到此为止，等待你的确认。
