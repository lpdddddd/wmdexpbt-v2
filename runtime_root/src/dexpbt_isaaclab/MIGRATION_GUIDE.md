# Offline Migration Guide (DexPBT IsaacLab)

This guide migrates the current experiment stack to an air-gapped GPU machine.

## 1) On source machine (with internet)

```bash
cd /root/autodl-tmp/dexpbt_isaaclab
chmod +x scripts/migration/*.sh
bash scripts/migration/bundle_offline.sh /root/autodl-tmp/dexpbt_isaaclab /root/autodl-tmp/dexpbt_isaaclab/offline_bundle isaaclab311
```

Optional: copy large checkpoints and datasets into `offline_bundle/artifacts/`.

## 2) Transfer bundle to shared disk

Copy `/root/autodl-tmp/dexpbt_isaaclab/offline_bundle` to target shared path (example `/shared/offline_bundle`).

## 3) On target GPU node (air-gapped)

```bash
mkdir -p /shared/install_tmp
tar -xzf /shared/offline_bundle/code/dexpbt_isaaclab_code.tar.gz -C /shared/install_tmp
cd /shared/install_tmp
chmod +x scripts/migration/*.sh
bash scripts/migration/install_offline_gpu.sh /shared/offline_bundle /shared/dexpbt_runtime
```

## 4) Smoke validation

```bash
cd /shared/dexpbt_runtime/src
bash scripts/migration/smoke_validate.sh /shared/dexpbt_runtime /shared/dexpbt_runtime/smoke_run
```

Run the printed training command once all runtime assets are confirmed.

## 5) Production run

Use the same command template from:
- `offline_bundle/MIGRATION_MANIFEST.md`
- `scripts/migration/smoke_validate.sh`

## Notes

- If Docker daemon is unavailable, this workflow does not depend on Docker.
- If `pip --no-index` fails, wheelhouse is incomplete; regenerate on source machine.
- For large model/checkpoint assets, store them in shared disk and use symlinks.
