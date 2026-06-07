# 完整迁移到 CPU 共享盘（代码 + 环境）

之前的 `offline_bundle.tar.gz` **只有 pip 轮子 + 代码**，**没有** `isaaclab311` conda 环境，且你遇到的 `Unexpected EOF` 说明 **tar.gz 没传完整**（源机完整 bundle 约 **5GB**，255MB 一定是坏的）。

本方案把训练真正需要的内容一次性打到 **`full_migration/`**，拷到 CPU 共享盘后，GPU 节点解压安装。

## 源机打包（有网、在本机执行）

```bash
cd /root/autodl-tmp/dexpbt_isaaclab
chmod +x scripts/migration/*.sh

# 输出目录约 15~20GB（conda 12G + 代码 + IsaacLab + Vulkan）
bash scripts/migration/bundle_full_to_shared.sh /root/autodl-tmp/dexpbt_full_migration

# 若传输不稳定，可分卷（每片 4GB）：
# SPLIT_GB=4 bash scripts/migration/bundle_full_to_shared.sh /root/autodl-tmp/dexpbt_full_migration
```

打包完成后自检：

```bash
gzip -t /root/autodl-tmp/dexpbt_full_migration/conda/isaaclab311.tar.gz
sha256sum /root/autodl-tmp/dexpbt_full_migration/conda/isaaclab311.tar.gz
du -sh /root/autodl-tmp/dexpbt_full_migration/*
```

## 传到 CPU 共享盘

### 方式 A：一键直传（推荐，可断点续传）

源机需能 `ssh` 到 CPU 节点（配置好密钥）：

```bash
cd /root/autodl-tmp/dexpbt_isaaclab
chmod +x scripts/migration/push_full_to_cpu.sh

# 改成你的 CPU 账号和共享盘路径
bash scripts/migration/push_full_to_cpu.sh \
  root@chchen-gpu-4-0:/volume/chchen/Pd/wmdexpbt/full_migration
```

网络不稳时分卷再传：

```bash
SPLIT_GB=4 bash scripts/migration/push_full_to_cpu.sh \
  root@chchen-gpu-4-0:/volume/chchen/Pd/wmdexpbt/full_migration
```

已打包过、只重传：

```bash
SKIP_BUILD=1 bash scripts/migration/push_full_to_cpu.sh \
  root@chchen-gpu-4-0:/volume/chchen/Pd/wmdexpbt/full_migration
```

### 方式 B：手动 rsync

```bash
rsync -avP --partial /root/autodl-tmp/dexpbt_full_migration/ \
  root@chchen-gpu-4-0:/volume/chchen/Pd/wmdexpbt/full_migration/
```

### 方式 C：源机无法直连 CPU 时

1. 先 `bundle_full_to_shared.sh` 打到本地目录  
2. 用网盘/内网跳板/运维提供的上传通道拷到 CPU 共享盘  
3. **不要**只传一个 `offline_bundle.tar.gz`（易损坏且不含 conda 环境）

传完后在 **CPU 或 GPU 节点** 校验：

```bash
cd /volume/chchen/Pd/wmdexpbt/full_migration
sha256sum -c SHA256SUMS
```

## 在共享盘上安装（建议在 GPU 节点执行一次）

```bash
cd /volume/chchen/Pd/wmdexpbt/full_migration
bash scripts/install_full_from_shared.sh \
  /volume/chchen/Pd/wmdexpbt/full_migration \
  /volume/chchen/Pd/wmdexpbt/runtime_root
```

## 开训前

```bash
source /volume/chchen/Pd/wmdexpbt/runtime_root/scripts/env_dexpbt.sh
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
cd "$DEXPBT_ROOT"
# 用 run_train_supervisor.sh，RUN_DIR 放在共享盘即可
```

## 说明

| 组件 | 是否包含 |
|------|----------|
| dexpbt 代码（不含 results） | ✅ |
| IsaacLab 源码 | ✅ |
| conda 环境 isaaclab311 | ✅（conda-pack） |
| Vulkan + EGL ICD | ✅ |
| 完整 results/ 历史实验 | ❌ 需单独 rsync |
| Isaac Sim 系统级安装 | ⚠️ 一般在 conda 环境里；若仍缺，需对照 Isaac 文档补装 |

CPU 开发机可以**只负责收文件**；真正 `torch.cuda` 和训练请在 **GPU 节点** 上验证。
