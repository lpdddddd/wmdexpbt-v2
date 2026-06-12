"""PBT Elite Broadcast for EIR-MPC (Phase D).

When a worker becomes a strong recovery performer its elite recovery memory
(bank top entries), gated residual head and action scorer should spread to the
weakest / collapsed workers instead of being lost when PBT only copies actor
weights. This module is the *testable core* of that mechanism, kept independent
of the heavy ``DexPBTPbtAlgoObserver`` so it can be unit-tested in isolation:

- ``should_broadcast`` / ``select_broadcast_targets``: trigger + target policy
  selection (bottom-k by nsucc_mean, excluding the source).
- ``write_broadcast`` / ``read_broadcast`` / ``mark_consumed``: a disk mailbox at
  ``<workspace>/eir_broadcast/<target:03d>.{json,payload.pt}`` mirroring the
  existing ``wm_imagined`` mailbox pattern. Consumption is idempotent and guarded
  by a monotonic ``frame`` so the same payload is never merged twice
  (constraint 19.4).
- ``decide_consume_mode``: live in-process merge when the running policy exposes
  an EIR bridge, else a forced-restart hand-off (the new process consumes the
  mailbox on startup).
- ``merge_payload_into_bridge``: priority-capped bank merge + residual / scorer
  copy into a live ``EIRBridge``.
- ``append_broadcast_event``: JSONL audit trail at
  ``metrics/eir_broadcast_events.jsonl``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch


CONSUME_LIVE = "live"
CONSUME_RESTART = "restart"


@dataclass
class EliteBroadcastConfig:
    enabled: bool = False
    threshold_nsucc_mean: float = 38.0
    gap_over_pop_mean: float = 6.0
    bottom_k: int = 3
    max_bank_merge_entries: int = 50_000
    top_entries: int = 20_000
    copy_residual_head: bool = True
    copy_scorer: bool = True
    freeze_elite_curriculum_frames: int = 100_000_000
    # never broadcast a near-empty bank (no useful memory to spread)
    min_source_bank_size: int = 100
    min_frames_between_broadcasts_per_source: int = 100_000_000
    min_frames_between_broadcasts_per_target: int = 100_000_000


def should_broadcast(
    source_nsucc_mean: float,
    pop_mean: float,
    *,
    cfg: EliteBroadcastConfig,
    source_bank_size: int = 0,
) -> bool:
    """True when this worker qualifies as an elite broadcast source."""
    if not cfg.enabled:
        return False
    if int(source_bank_size) < int(cfg.min_source_bank_size):
        return False
    if float(source_nsucc_mean) >= float(cfg.threshold_nsucc_mean):
        return True
    if float(source_nsucc_mean) >= float(pop_mean) + float(cfg.gap_over_pop_mean):
        return True
    return False


def select_broadcast_targets(
    source_idx: int,
    pop_nsucc: dict[int, float],
    *,
    cfg: EliteBroadcastConfig,
    exclude: Optional[set[int]] = None,
) -> list[int]:
    """Bottom-k policies by nsucc_mean, excluding the source / excluded set."""
    exclude = set(exclude or set())
    exclude.add(int(source_idx))
    cands = [(idx, float(v)) for idx, v in pop_nsucc.items() if int(idx) not in exclude]
    cands.sort(key=lambda kv: kv[1])  # ascending: weakest first
    k = max(0, int(cfg.bottom_k))
    return [int(idx) for idx, _ in cands[:k]]


def _mailbox_dir(workspace: Path | str) -> Path:
    return Path(workspace) / "eir_broadcast"


def _meta_path(workspace: Path | str, target_idx: int) -> Path:
    return _mailbox_dir(workspace) / f"{int(target_idx):03d}.json"


def _payload_path(workspace: Path | str, target_idx: int) -> Path:
    return _mailbox_dir(workspace) / f"{int(target_idx):03d}.payload.pt"


def _consumed_path(workspace: Path | str, target_idx: int) -> Path:
    return _mailbox_dir(workspace) / f"{int(target_idx):03d}.consumed.json"


def _last_write_path(workspace: Path | str, target_idx: int) -> Path:
    return _mailbox_dir(workspace) / f"{int(target_idx):03d}.last_write.json"


def read_target_last_broadcast_frame(workspace: Path | str, target_idx: int) -> int:
    """Return the env-frame of the last broadcast write to ``target_idx``."""
    p = _last_write_path(workspace, target_idx)
    if not p.exists():
        return -1
    try:
        return int(json.loads(p.read_text()).get("frame", -1))
    except Exception:
        return -1


def source_broadcast_allowed(
    last_source_broadcast_frame: int,
    current_frame: int,
    *,
    cfg: EliteBroadcastConfig,
) -> bool:
    """True when this source may broadcast again (per-source cooldown)."""
    if int(last_source_broadcast_frame) < 0:
        return True
    return (
        int(current_frame) - int(last_source_broadcast_frame)
        >= int(cfg.min_frames_between_broadcasts_per_source)
    )


def target_broadcast_allowed(
    workspace: Path | str,
    target_idx: int,
    current_frame: int,
    *,
    cfg: EliteBroadcastConfig,
) -> bool:
    """True when ``target_idx`` may receive a new payload (per-target cooldown)."""
    last = read_target_last_broadcast_frame(workspace, target_idx)
    if int(last) < 0:
        return True
    return (
        int(current_frame) - int(last)
        >= int(cfg.min_frames_between_broadcasts_per_target)
    )


def record_target_broadcast_write(
    workspace: Path | str,
    target_idx: int,
    *,
    frame: int,
    source_policy: int,
) -> None:
    """Persist the last broadcast write time for per-target cooldown."""
    d = _mailbox_dir(workspace)
    d.mkdir(parents=True, exist_ok=True)
    p = _last_write_path(workspace, target_idx)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"frame": int(frame), "source_policy": int(source_policy)}))
    os.replace(tmp, p)


def build_broadcast_payload(
    bridge: Any,
    *,
    cfg: EliteBroadcastConfig,
    source_idx: int,
    source_nsucc_mean: float,
    source_nsucc_max: float,
    frame: int,
) -> dict[str, Any]:
    """Assemble a broadcast payload from a live ``EIRBridge``.

    Holds top-N bank entries (priority-capped), the residual head + scorer state
    dicts and the scorer warmup counter so a recovering target inherits real
    memory rather than a fresh head.
    """
    payload: dict[str, Any] = {
        "source_policy": int(source_idx),
        "source_nsucc_mean": float(source_nsucc_mean),
        "source_nsucc_max": float(source_nsucc_max),
        "frame": int(frame),
        "bank_entries": 0,
    }
    bank = getattr(bridge, "bank", None)
    if bank is not None and getattr(bank, "size", 0) > 0:
        top = bank.top_entries_state(int(cfg.max_bank_merge_entries))
        payload["bank_state"] = top
        payload["bank_entries"] = int(top.get("size", 0))
    if cfg.copy_residual_head:
        rb = getattr(bridge, "residual_bridge", None)
        head = getattr(rb, "head", None) if rb is not None else None
        if head is not None:
            payload["residual_head"] = {k: v.detach().cpu() for k, v in head.state_dict().items()}
    if cfg.copy_scorer:
        scorer = getattr(bridge, "scorer", None)
        if scorer is not None:
            payload["scorer"] = {k: v.detach().cpu() for k, v in scorer.state_dict().items()}
            payload["scorer_warmup_iters_done"] = int(
                getattr(bridge, "scorer_warmup_iters_done", 0)
            )
    return payload


def write_broadcast(
    workspace: Path | str,
    target_idx: int,
    payload: dict[str, Any],
) -> Path:
    """Atomically write a broadcast payload to the target's mailbox slot.

    The ``.payload.pt`` carries tensors; the ``.json`` is a light marker used to
    cheaply detect/poll a pending payload. Both are written tmp-then-rename so a
    concurrent reader never sees a half-written file.
    """
    d = _mailbox_dir(workspace)
    d.mkdir(parents=True, exist_ok=True)
    payload_path = _payload_path(workspace, target_idx)
    meta_path = _meta_path(workspace, target_idx)
    tmp_payload = payload_path.with_suffix(".pt.tmp")
    tmp_meta = meta_path.with_suffix(".json.tmp")
    torch.save(payload, tmp_payload)
    os.replace(tmp_payload, payload_path)
    meta = {
        "source_policy": int(payload.get("source_policy", -1)),
        "target_policy": int(target_idx),
        "frame": int(payload.get("frame", 0)),
        "bank_entries": int(payload.get("bank_entries", 0)),
        "has_residual": "residual_head" in payload,
        "has_scorer": "scorer" in payload,
        "payload_path": str(payload_path.resolve()),
    }
    tmp_meta.write_text(json.dumps(meta))
    os.replace(tmp_meta, meta_path)
    record_target_broadcast_write(
        workspace,
        target_idx,
        frame=int(payload.get("frame", 0)),
        source_policy=int(payload.get("source_policy", -1)),
    )
    return meta_path


def _last_consumed_frame(workspace: Path | str, target_idx: int) -> int:
    p = _consumed_path(workspace, target_idx)
    if not p.exists():
        return -1
    try:
        return int(json.loads(p.read_text()).get("frame", -1))
    except Exception:
        return -1


def read_broadcast(
    workspace: Path | str,
    target_idx: int,
) -> Optional[dict[str, Any]]:
    """Return a pending payload meta dict for ``target_idx`` or None.

    A payload is *pending* only when its ``frame`` is newer than the last
    consumed frame, guaranteeing idempotent consumption.
    """
    meta_path = _meta_path(workspace, target_idx)
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return None
    if int(meta.get("frame", 0)) <= _last_consumed_frame(workspace, target_idx):
        return None
    return meta


def load_payload(meta: dict[str, Any]) -> Optional[dict[str, Any]]:
    path = meta.get("payload_path")
    if not path or not Path(path).exists():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None


def mark_consumed(
    workspace: Path | str,
    target_idx: int,
    *,
    frame: int,
    source_policy: int,
    mode: str,
    merged_entries: int,
) -> None:
    """Record consumption (idempotency guard) and clean up the mailbox slot."""
    d = _mailbox_dir(workspace)
    d.mkdir(parents=True, exist_ok=True)
    rec = {
        "frame": int(frame),
        "source_policy": int(source_policy),
        "mode": str(mode),
        "merged_entries": int(merged_entries),
    }
    tmp = _consumed_path(workspace, target_idx).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, _consumed_path(workspace, target_idx))
    # remove the pending payload + meta so it is not re-read
    for p in (_payload_path(workspace, target_idx), _meta_path(workspace, target_idx)):
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass


def decide_consume_mode(bridge_available: bool) -> str:
    """Live in-process merge when an EIR bridge exists, else forced-restart."""
    return CONSUME_LIVE if bridge_available else CONSUME_RESTART


def merge_payload_into_bridge(
    bridge: Any,
    payload: dict[str, Any],
    *,
    cfg: EliteBroadcastConfig,
) -> dict[str, Any]:
    """Live-merge a broadcast payload into a running ``EIRBridge``.

    Returns a summary dict ``{merged_entries, residual_copied, scorer_copied}``.
    Bank entries are folded in priority-capped (never replacing the target's own
    accumulated memory); the residual head + scorer are copied wholesale.
    """
    out = {"merged_entries": 0, "residual_copied": False, "scorer_copied": False}
    bank = getattr(bridge, "bank", None)
    bank_state = payload.get("bank_state")
    if bank is not None and isinstance(bank_state, dict):
        try:
            out["merged_entries"] = int(
                bank.merge(bank_state, max_entries=int(cfg.max_bank_merge_entries))
            )
        except Exception:
            out["merged_entries"] = 0
    if cfg.copy_residual_head:
        rb = getattr(bridge, "residual_bridge", None)
        head = getattr(rb, "head", None) if rb is not None else None
        rsd = payload.get("residual_head")
        if head is not None and isinstance(rsd, dict):
            try:
                head.load_state_dict(rsd, strict=False)
                out["residual_copied"] = True
            except Exception:
                out["residual_copied"] = False
    if cfg.copy_scorer:
        scorer = getattr(bridge, "scorer", None)
        ssd = payload.get("scorer")
        if scorer is not None and isinstance(ssd, dict):
            try:
                scorer.load_state_dict(ssd, strict=False)
                warm = payload.get("scorer_warmup_iters_done")
                if warm is not None:
                    bridge.scorer_warmup_iters_done = max(
                        int(getattr(bridge, "scorer_warmup_iters_done", 0)), int(warm)
                    )
                out["scorer_copied"] = True
            except Exception:
                out["scorer_copied"] = False
    return out


def append_broadcast_event(metrics_dir: Path | str, event: dict[str, Any]) -> None:
    """Append one broadcast/consume event to ``eir_broadcast_events.jsonl``."""
    d = Path(metrics_dir)
    try:
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "eir_broadcast_events.jsonl", "a") as f:
            f.write(json.dumps(event) + "\n")
    except OSError:
        pass
