"""v17.2 hardened PBT control-plane: clocks, snapshot, restore, skill_delta gates."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
    RollingNsuccWindow,
    SKILL_DELTA_IMPROVE_EPS,
    SKILL_DELTA_STABLE_POS_EPS,
    SKILL_DELTA_WINDOW,
    compute_skill_delta,
)

PBT_SNAPSHOT_SCHEMA_VERSION = 2
PBT_SNAPSHOT_REQUIRED_MEMBERS = ("model.pth", "meta.json", "MANIFEST")
PBT_SNAPSHOT_HASHED_MEMBERS = ("model.pth", "meta.json")
PBT_STUCK_NSUCC_MEAN = 1.0
PBT_PROGRESS_INHERIT_CONTINUE_TARGET = "continue_target_clock"


@dataclass
class PbtClockState:
    population_frame: int
    lineage_id: str
    lineage_start_pf: int
    weight_age_frame: int


@dataclass
class PbtOptimizerSnapshot:
    epoch: int = 0
    optimizer_step: int = 0
    adam_moments_present: bool = False


@dataclass
class PbtCopyEvent:
    source_policy_id: int = -1
    source_population_frame: int = 0
    target_population_frame: int = 0
    progress_inheritance_mode: str = PBT_PROGRESS_INHERIT_CONTINUE_TARGET


@dataclass
class PbtStateSnapshot:
    schema_version: int = PBT_SNAPSHOT_SCHEMA_VERSION
    clocks: PbtClockState = field(default_factory=lambda: PbtClockState(0, "", 0, 0))
    optimizer_state: PbtOptimizerSnapshot = field(default_factory=PbtOptimizerSnapshot)
    copy_event: PbtCopyEvent | None = None

    def to_meta_fields(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": int(self.schema_version),
            "population_frame": int(self.clocks.population_frame),
            "lineage_id": str(self.clocks.lineage_id),
            "lineage_start_pf": int(self.clocks.lineage_start_pf),
            "weight_age_frame": int(self.clocks.weight_age_frame),
            "optimizer_epoch": int(self.optimizer_state.epoch),
            "optimizer_step": int(self.optimizer_state.optimizer_step),
            "adam_moments_present": bool(self.optimizer_state.adam_moments_present),
        }
        if self.copy_event is not None:
            out["copy_event"] = asdict(self.copy_event)
        return out

    @classmethod
    def from_meta(cls, meta: dict[str, Any] | None) -> PbtStateSnapshot | None:
        if not meta:
            return None
        pf = meta.get("population_frame", meta.get("frame"))
        lid = meta.get("lineage_id")
        if pf is None or lid is None:
            return None
        lineage_start = meta.get("lineage_start_pf")
        if lineage_start is None:
            lineage_start = int(pf)
        weight_age = meta.get("weight_age_frame", meta.get("frame", 0))
        copy_raw = meta.get("copy_event")
        copy_event = None
        if isinstance(copy_raw, dict):
            copy_event = PbtCopyEvent(
                source_policy_id=int(copy_raw.get("source_policy_id", -1)),
                source_population_frame=int(copy_raw.get("source_population_frame", 0)),
                target_population_frame=int(copy_raw.get("target_population_frame", 0)),
                progress_inheritance_mode=str(
                    copy_raw.get("progress_inheritance_mode", PBT_PROGRESS_INHERIT_CONTINUE_TARGET)
                ),
            )
        return cls(
            clocks=PbtClockState(
                population_frame=int(pf),
                lineage_id=str(lid),
                lineage_start_pf=int(lineage_start),
                weight_age_frame=int(weight_age),
            ),
            optimizer_state=PbtOptimizerSnapshot(
                epoch=int(meta.get("optimizer_epoch", meta.get("epoch", 0)) or 0),
                optimizer_step=int(meta.get("optimizer_step", 0) or 0),
                adam_moments_present=bool(meta.get("adam_moments_present", False)),
            ),
            copy_event=copy_event,
        )


def derive_pbt_iteration(population_frame: int, tick: int) -> int:
    tick_i = max(int(tick), 1)
    return int(population_frame) // tick_i


def pbt_raw_training_frame(algo) -> int:
    """v17.2 debug hotfix: use live algo.frame for tick scheduling between commits."""
    return int(getattr(algo, "frame", 0) or 0)


def evaluate_pbt_tick_schedule(
    *,
    raw_frame: int,
    self_pbt_iteration: int,
    last_mutation_frame: int,
    start_after: int,
    initial_delay: int,
    interval_steps: int,
) -> tuple[bool, int, str]:
    """Return (should_tick, next_pbt_iteration, skip_reason).

    Scheduling uses raw training frames so a prior LATEST.commit cannot freeze
    progress at the last snapshot population_frame.

    Tick eligibility is frame-based (``raw_frame - last_mutation_frame``) so
    consolidation stages can widen ``interval_steps`` without shrinking
    ``derive_pbt_iteration`` below ``self_pbt_iteration`` and blocking forever.
    """
    derived = derive_pbt_iteration(int(raw_frame), int(interval_steps))
    if int(raw_frame) < int(start_after):
        return False, derived, "before_start_after"
    if int(raw_frame) - int(last_mutation_frame) < int(initial_delay):
        return False, derived, "initial_delay"
    frames_since_last_tick = int(raw_frame) - int(last_mutation_frame)
    if frames_since_last_tick < int(interval_steps):
        return False, derived, "interval_not_elapsed"
    next_iteration = max(int(self_pbt_iteration) + 1, derived)
    return True, next_iteration, ""


def read_latest_commit_info(policy_workspace: Path) -> dict[str, Any]:
    latest_path = policy_workspace / "LATEST.commit"
    if not latest_path.is_file():
        return {
            "latest_commit_exists": False,
            "latest_commit_iteration": None,
            "latest_commit_population_frame": None,
        }
    try:
        latest = json.loads(latest_path.read_text())
    except Exception:
        return {
            "latest_commit_exists": False,
            "latest_commit_iteration": None,
            "latest_commit_population_frame": None,
        }
    commit_dir_name = str(latest.get("commit_dir") or "")
    commit_dir = policy_workspace / commit_dir_name if commit_dir_name else None
    iteration = None
    population_frame = None
    if commit_dir is not None and commit_dir.is_dir():
        manifest_path = commit_dir / "MANIFEST"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text())
                iteration = manifest.get("pbt_iteration")
                population_frame = manifest.get("population_frame")
            except Exception:
                pass
    return {
        "latest_commit_exists": True,
        "latest_commit_iteration": iteration,
        "latest_commit_population_frame": population_frame,
    }


def restore_pbt_tick_state_from_workspace(
    *,
    policy_workspace: Path,
    self_pbt_iteration: int,
    last_mutation_frame: int,
    last_mutation_frame_resume: int | None = None,
) -> tuple[int, int]:
    """Restore PBT tick counters after a supervisor-driven train.py relaunch.

    PBT restart manifests already forward ``*_resume`` CLI args; this path
    covers crash/supervisor reloads that only pass ``--checkpoint``.
    """
    iteration = int(self_pbt_iteration)
    last_frame = int(last_mutation_frame)
    if last_mutation_frame_resume is not None:
        last_frame = int(last_mutation_frame_resume)
    else:
        last_frame = 0
    latest = read_latest_commit_info(policy_workspace)
    if latest.get("latest_commit_exists"):
        commit_iter = latest.get("latest_commit_iteration")
        commit_frame = latest.get("latest_commit_population_frame")
        if commit_iter is not None and (
            iteration < 0 or int(commit_iter) >= iteration
        ):
            iteration = int(commit_iter)
        if commit_frame is not None and int(commit_frame) >= int(last_frame):
            last_frame = int(commit_frame)
    return iteration, last_frame


def derive_lineage_frame(population_frame: int, lineage_start_pf: int) -> int:
    return max(0, int(population_frame) - int(lineage_start_pf))


def quantize_population_frame(raw_frame: int, tick: int) -> int:
    tick_i = max(int(tick), 1)
    return (int(raw_frame) // tick_i) * tick_i


def effective_population_frame_target(target_frame: int, tick: int) -> int:
    """Ceil target onto the next authoritative tick so we never stop early.

    5B / 20M already divides evenly; non-multiples must round *up*.
    """
    if int(target_frame) <= 0:
        return 0
    tick_i = max(int(tick), 1)
    t = int(target_frame)
    return ((t + tick_i - 1) // tick_i) * tick_i


def population_frame_target_reached(population_frame: int, target_frame: int, tick: int) -> bool:
    """Compare PBT clocks at ceil-quantized target resolution."""
    if int(target_frame) <= 0:
        return False
    return int(population_frame) >= effective_population_frame_target(target_frame, tick)


def pbt_population_frame(algo) -> int:
    if getattr(algo, "_pbt_restored_from_pbt_copy", False) and not getattr(
        algo, "_pbt_frame_snapshot_driven", False
    ):
        raise RuntimeError("PBT_FRAME_MUST_BE_SNAPSHOT_DRIVEN")
    pf = getattr(algo, "_pbt_population_frame", None)
    if pf is not None:
        return int(pf)
    return int(getattr(algo, "frame", 0) or 0)


def pbt_scheduler_sync_clock(
    algo,
    *,
    tick: int,
    lineage_id: str,
    lineage_start_pf: int,
    weight_age_frame: int,
) -> int:
    """PBT scheduler: sole writer of population_frame at tick boundary."""
    raw = int(getattr(algo, "frame", 0) or 0)
    pf = quantize_population_frame(raw, tick)
    algo.frame = pf
    algo._pbt_population_frame = pf
    algo._pbt_lineage_id = str(lineage_id)
    algo._pbt_lineage_start_pf = int(lineage_start_pf)
    algo._pbt_weight_age_frame = int(weight_age_frame)
    return pf


def _frames_per_epoch(agent) -> int:
    hor_len = int(getattr(agent, "horizon_length", 0) or 0)
    num_actors = int(getattr(agent, "num_actors", 0) or 0)
    return max(hor_len * num_actors, 0)


def _set_agent_epoch_from_population_frame(agent, population_frame: int, *, fallback_epoch: int | None) -> None:
    fpe = _frames_per_epoch(agent)
    if fpe > 0:
        agent.epoch_num = int(population_frame) // int(fpe)
    elif fallback_epoch is not None:
        agent.epoch_num = int(fallback_epoch)


def apply_pbt_state_snapshot(agent, snap: PbtStateSnapshot) -> None:
    pf = int(snap.clocks.population_frame)
    agent.frame = pf
    agent._pbt_population_frame = pf
    agent._pbt_lineage_id = str(snap.clocks.lineage_id)
    agent._pbt_lineage_start_pf = int(snap.clocks.lineage_start_pf)
    agent._pbt_weight_age_frame = int(snap.clocks.weight_age_frame)
    agent._pbt_frame_snapshot_driven = True
    _set_agent_epoch_from_population_frame(
        agent, pf, fallback_epoch=snap.optimizer_state.epoch
    )
    agent._pbt_optimizer_step = int(snap.optimizer_state.optimizer_step)
    agent._pbt_adam_moments_present = bool(snap.optimizer_state.adam_moments_present)


def build_copy_snapshot(
    *,
    target_population_frame: int,
    source_population_frame: int,
    source_policy_id: int,
    new_lineage_id: str,
    source_weight_age: int,
) -> PbtStateSnapshot:
    return PbtStateSnapshot(
        clocks=PbtClockState(
            population_frame=int(target_population_frame),
            lineage_id=str(new_lineage_id),
            lineage_start_pf=int(target_population_frame),
            weight_age_frame=int(source_weight_age),
        ),
        copy_event=PbtCopyEvent(
            source_policy_id=int(source_policy_id),
            source_population_frame=int(source_population_frame),
            target_population_frame=int(target_population_frame),
            progress_inheritance_mode=PBT_PROGRESS_INHERIT_CONTINUE_TARGET,
        ),
    )


def load_meta_for_checkpoint(checkpoint_path: str | Path) -> dict[str, Any] | None:
    ckpt = Path(checkpoint_path)
    meta_path = ckpt.with_suffix(".json")
    if not meta_path.is_file():
        return None
    try:
        data = json.loads(meta_path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def pbt_pure_restore_clocks(
    agent,
    ckpt: dict[str, Any],
    *,
    meta: dict[str, Any] | None,
    target_frame_before_restore: int | None,
    restart_source_policy: int | None,
) -> PbtStateSnapshot:
    """Deterministic clock restore. No gap heuristics."""
    if restart_source_policy is not None and target_frame_before_restore is None:
        raise RuntimeError("PBT_FRAME_MUST_BE_SNAPSHOT_DRIVEN")

    meta_snap = PbtStateSnapshot.from_meta(meta)
    source_frame = int(ckpt.get("frame", 0) or 0)
    source_weight_age = int(source_frame)
    if meta is not None:
        waf = meta.get("weight_age_frame")
        if waf is not None:
            source_weight_age = int(waf)

    if target_frame_before_restore is not None:
        target_pf = int(target_frame_before_restore)
        new_lineage = str(getattr(agent, "_pbt_lineage_id_resume", "") or "")
        if not new_lineage:
            import uuid

            new_lineage = str(uuid.uuid4())
        snap = build_copy_snapshot(
            target_population_frame=target_pf,
            source_population_frame=source_frame,
            source_policy_id=int(restart_source_policy or -1),
            new_lineage_id=new_lineage,
            source_weight_age=source_weight_age,
        )
        agent._pbt_restored_from_pbt_copy = True
        apply_pbt_state_snapshot(agent, snap)
        agent._pbt_copy_source_frame = float(source_frame)
        agent._pbt_copy_target_frame_before_restore = float(target_pf)
        agent._pbt_copy_reset_progress_due_to_frame_gap = 0.0
        print(
            "[PBT] restore: pure copy snapshot "
            f"target_pf={target_pf} source_pf={source_frame} lineage={new_lineage}"
        )
        return snap

    if meta_snap is not None:
        apply_pbt_state_snapshot(agent, meta_snap)
        print(
            "[PBT] restore: meta snapshot population_frame="
            f"{meta_snap.clocks.population_frame} lineage={meta_snap.clocks.lineage_id}"
        )
        return meta_snap

    pf = int(source_frame)
    import uuid

    snap = PbtStateSnapshot(
        clocks=PbtClockState(
            population_frame=pf,
            lineage_id=str(uuid.uuid4()),
            lineage_start_pf=pf,
            weight_age_frame=pf,
        ),
        optimizer_state=PbtOptimizerSnapshot(
            epoch=int(ckpt.get("epoch", 0) or 0),
        ),
    )
    apply_pbt_state_snapshot(agent, snap)
    agent._pbt_frame_snapshot_driven = False
    if "epoch" in ckpt:
        agent.epoch_num = int(ckpt["epoch"])
    if "frame" in ckpt:
        agent.frame = int(ckpt["frame"])
        agent._pbt_population_frame = int(ckpt["frame"])
    print(f"[PBT] restore: legacy checkpoint clocks frame={pf}")
    return snap


def assert_pbt_clock_invariants(agent, *, tick: int) -> None:
    pf = pbt_population_frame(agent)
    tick_i = max(int(tick), 1)
    stored_pf = getattr(agent, "_pbt_population_frame", None)
    if stored_pf is not None and int(stored_pf) != int(getattr(agent, "frame", 0)):
        raise RuntimeError("PBT_CLOCK_DESYNC:population_frame!=agent.frame")
    iteration_derived = derive_pbt_iteration(pf, tick_i)
    stored_iter = getattr(agent, "_pbt_derived_iteration", None)
    if stored_iter is not None and int(stored_iter) != iteration_derived:
        raise RuntimeError("PBT_CLOCK_DESYNC:derived_iteration_mismatch")


def skill_delta_for_policy(
    window: RollingNsuccWindow | None,
    *,
    interval_steps: int,
    window_len: int = SKILL_DELTA_WINDOW,
) -> float | None:
    return compute_skill_delta(
        window, window_len=window_len, interval_steps=interval_steps
    )


def skill_delta_improving(delta: float | None) -> bool:
    return delta is not None and float(delta) >= float(SKILL_DELTA_IMPROVE_EPS)


def skill_delta_source_qualified(delta: float | None) -> bool:
    return delta is not None and float(delta) >= float(SKILL_DELTA_STABLE_POS_EPS)


def skill_delta_stuck(
    window: RollingNsuccWindow | None,
    delta: float | None,
    *,
    stuck_nsucc_mean: float = PBT_STUCK_NSUCC_MEAN,
) -> bool:
    if delta is None or window is None or not window.samples:
        return False
    nm = float(window.samples[-1].nsucc_mean)
    return nm < float(stuck_nsucc_mean) and float(delta) <= 0.0


def pbt_copy_allowed_for_target(
    window: RollingNsuccWindow | None,
    *,
    interval_steps: int,
) -> tuple[bool, str, float | None]:
    delta = skill_delta_for_policy(window, interval_steps=interval_steps)
    if delta is None:
        return False, "skill_delta_invalid", None
    if skill_delta_improving(delta):
        return False, "skill_delta_improving", delta
    if not skill_delta_stuck(window, delta):
        return False, "not_stuck", delta
    return True, "ok", delta


def pbt_copy_allowed_for_adaptive_rescue_target(
    window: RollingNsuccWindow | None,
    *,
    interval_steps: int,
    hard_collapse: bool = False,
) -> tuple[bool, str, float | None]:
    """Target gate for adaptive retention/hard-collapse rescue only.

    Adaptive rescue is armed by retention degradation (or hard collapse), not by
    the generic stuck gate (``nsucc_mean < 1``).  A worker that fell from a high
    peak to a still-elevated mean must still be eligible for rescue.

    Semantics (caller already passed retention/cooldown/quota triggers):
    - hard collapse confirmed → allow source cascade
    - skill_delta clearly improving (≥ ``SKILL_DELTA_IMPROVE_EPS``) → reject
    - skill_delta ≤ 0, zero, or unavailable → allow (invalid delta does not
      override valid retention-degradation evidence)
    """
    delta = skill_delta_for_policy(window, interval_steps=interval_steps)
    if hard_collapse:
        return True, "adaptive_hard_collapse", delta
    if skill_delta_improving(delta):
        return False, "skill_delta_improving", delta
    if delta is None:
        return True, "skill_delta_unavailable_ok", None
    return True, "ok", delta


def pbt_copy_allowed_for_live_source(
    window: RollingNsuccWindow | None,
    *,
    interval_steps: int,
) -> tuple[bool, str]:
    delta = skill_delta_for_policy(window, interval_steps=interval_steps)
    if not skill_delta_source_qualified(delta):
        return False, "source_skill_delta_invalid_or_unstable"
    return True, "ok"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_snapshot_commit(staging_dir: Path, *, tick: int) -> tuple[bool, str]:
    manifest_path = staging_dir / "MANIFEST"
    if not manifest_path.is_file():
        return False, "no_manifest"
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return False, "manifest_corrupt"
    if not isinstance(manifest, dict):
        return False, "manifest_not_dict"
    member_shas = manifest.get("member_shas")
    if not isinstance(member_shas, dict):
        return False, "manifest_missing_member_shas"
    for name in PBT_SNAPSHOT_HASHED_MEMBERS:
        expected = member_shas.get(name)
        if expected is None:
            return False, f"missing_member:{name}"
        member_path = staging_dir / name
        if not member_path.is_file():
            return False, f"missing_file:{name}"
        if _sha256_file(member_path) != str(expected):
            return False, f"member_corrupt:{name}"
    if not manifest_path.is_file():
        return False, "missing_file:MANIFEST"
    if manifest.get("population_frame") is None:
        return False, "missing_population_frame"
    if manifest.get("lineage_id") is None:
        return False, "missing_lineage_id"
    pf = int(manifest["population_frame"])
    tick_i = max(int(tick), 1)
    if manifest.get("lineage_start_pf") is not None:
        lsp = int(manifest["lineage_start_pf"])
        if lsp % tick_i != 0:
            return False, "lineage_start_not_quantized"
        if lsp > pf:
            return False, "lineage_after_now"
    commit_id = manifest.get("commit_id")
    if not commit_id:
        return False, "missing_commit_id"
    if pf % tick_i != 0 and not manifest.get("allow_unquantized_clock", False):
        return False, "clock_not_quantized"
    return True, "ok"


def snapshot_commit_barrier(
    *,
    staging_dir: Path,
    model_src: Path,
    meta_src: Path,
    publish_model: Path,
    publish_meta: Path,
    manifest: dict[str, Any],
    sidecar_src: Path | None = None,
    tick: int,
) -> Path:
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_src, staging_dir / "model.pth")
    shutil.copy2(meta_src, staging_dir / "meta.json")
    member_shas = {
        "model.pth": _sha256_file(staging_dir / "model.pth"),
        "meta.json": _sha256_file(staging_dir / "meta.json"),
    }
    if sidecar_src is not None and sidecar_src.is_file():
        shutil.copy2(sidecar_src, staging_dir / "awr_sidecar.pt")
        member_shas["awr_sidecar.pt"] = _sha256_file(staging_dir / "awr_sidecar.pt")
    manifest = dict(manifest)
    manifest["schema_version"] = PBT_SNAPSHOT_SCHEMA_VERSION
    manifest["member_shas"] = member_shas
    commit_id = hashlib.sha256(
        json.dumps(member_shas, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    manifest["commit_id"] = commit_id
    manifest_path = staging_dir / "MANIFEST"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    ok, reason = validate_snapshot_commit(staging_dir, tick=tick)
    if not ok:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise RuntimeError(f"PBT_SNAPSHOT_COMMIT_FAILED:{reason}")
    publish_model_tmp = publish_model.with_suffix(publish_model.suffix + ".publish")
    publish_meta_tmp = publish_meta.with_suffix(publish_meta.suffix + ".publish")
    shutil.copy2(staging_dir / "model.pth", publish_model_tmp)
    shutil.copy2(staging_dir / "meta.json", publish_meta_tmp)
    publish_model_tmp.replace(publish_model)
    publish_meta_tmp.replace(publish_meta)
    commit_dir = staging_dir.parent / f"{staging_dir.name}.commit"
    if commit_dir.exists():
        shutil.rmtree(commit_dir)
    staging_dir.rename(commit_dir)
    latest = publish_meta.parent / "LATEST.commit"
    latest.write_text(json.dumps({"commit_id": commit_id, "commit_dir": commit_dir.name}))
    return commit_dir


def validate_published_meta(meta_path: Path, *, tick: int) -> None:
    commit_dir = meta_path.parent / f"{meta_path.stem}.commit"
    if commit_dir.is_dir():
        ok, reason = validate_snapshot_commit(commit_dir, tick=tick)
        if not ok:
            raise RuntimeError(f"PBT_SNAPSHOT_INVALID:{reason}")
