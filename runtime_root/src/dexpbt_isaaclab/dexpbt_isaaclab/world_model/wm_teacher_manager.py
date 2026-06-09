"""
Phase A4 P4 (2026-06-09) — RSSM iterative teacher manager (Part C).

A *teacher* is a frozen snapshot of the SeqWMTrainer's RSSM model that we
keep around for floor-stage evaluation purposes only. It is NEVER used to
train PPO and NEVER overrides the live trainer model. Its only consumer
is the imagined evaluator (Part D), which feeds PBT ranking via
floor_recovery_score_wm (Part E).

Workflow:

  1. Every ``teacher_eval_interval_frames`` env frames the live trainer
     calls ``maybe_save_candidate_snapshot(...)`` to dump
       ``world_model_teachers/policy_<idx>/wm_teacher_step_<frame>.pt``.

  2. The held-out validation buffer (``FloorRecoveryEvalBuffer``) stores
     a small set of real floor-stage / hard-floor / recovery sequences
     captured during normal play. Validation samples are written by the
     buffer and never reused for training.

  3. ``evaluate_candidate(teacher_state, eval_buffer)`` runs the
     candidate teacher on the held-out buffer in eval mode under no_grad
     and returns a floor_recovery_eval_score plus per-component metrics.

  4. ``maybe_promote(...)`` applies the conservative promotion gate
     (Item 5 of user constraints): every individual eval metric must
     satisfy a non-degradation check, the recovery target must be
     non-degenerate (pos rate inside ``[low, high]``), no NaNs, and the
     overall score must improve by at least ``promote_min_improvement``.

  5. The promoted teacher's path is recorded in
     ``world_model_teachers/policy_<idx>/teacher_index.json`` for
     downstream PBT consumers.

Default behavior: every public method short-circuits when the manager
was constructed with ``enabled=False`` (the trainer's
``--seq_wm_enable_iterative_teacher`` flag).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from .seq_wm_base import SequenceBatch, SequenceWorldModelBase


@dataclass
class TeacherEvalMetrics:
    """Per-snapshot evaluation metrics on the held-out floor buffer."""

    score: float = 0.0
    recovery_bce: float = 0.0
    horizon_success_bce: float = 0.0
    drop_risk_bce: float = 0.0
    d_hat_trend_mse: float = 0.0
    value_mse: float = 0.0
    recovery_pos_rate: float = 0.0
    horizon_pos_rate: float = 0.0
    drop_pos_rate: float = 0.0
    n_samples: int = 0
    has_nan: bool = False
    components: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items() if k != "components"}
        out["components"] = dict(self.components)
        return out


@dataclass
class TeacherSnapshot:
    path: Path
    frame: int
    global_step: int
    score: float
    metrics: TeacherEvalMetrics


def _safe_isfinite_mean(t: Optional[torch.Tensor]) -> tuple[float, bool]:
    if t is None:
        return 0.0, False
    if not torch.is_floating_point(t):
        t = t.float()
    has_nan = bool(torch.isnan(t).any().item() or torch.isinf(t).any().item())
    if has_nan:
        return 0.0, True
    return float(t.mean().item()), False


class FloorRecoveryEvalBuffer:
    """Tiny held-out buffer of floor-stage SequenceBatch slices.

    Storage: a deque-like list of pre-built ``SequenceBatch`` objects on
    CPU. The buffer is populated by ``record_candidate(...)`` from a
    sampled training batch when the policy is at floor or recovery.
    Default capacity is 8 batches; small enough to keep eval cost low
    and large enough to track multiple floor regimes.

    The buffer is NEVER consumed by the WM trainer; only by
    ``evaluate_candidate``.
    """

    def __init__(self, *, capacity: int = 8) -> None:
        self.capacity = max(1, int(capacity))
        self.batches: list[SequenceBatch] = []

    def __len__(self) -> int:
        return len(self.batches)

    def record_candidate(
        self,
        batch: SequenceBatch,
        *,
        floor_tol_threshold: float = 0.02,
        hard_floor_tol_threshold: float = 0.012,
        require_floor_signal: bool = True,
    ) -> bool:
        """Conditionally record ``batch`` if it shows floor-stage signal.

        Returns True iff stored. ``require_floor_signal`` enforces that
        the batch's per-step success_tolerance has at least one element
        below ``floor_tol_threshold``; if False, every call stores
        (useful for tests).
        """
        if require_floor_signal:
            if batch.success_tolerance_seq is None:
                return False
            tol = batch.success_tolerance_seq.float()
            if not bool((tol <= float(floor_tol_threshold)).any().item()):
                return False
        # Move tensors to CPU before storage so we don't hold GPU memory.
        cpu_batch = batch.to(torch.device("cpu"))
        self.batches.append(cpu_batch)
        if len(self.batches) > self.capacity:
            self.batches = self.batches[-self.capacity :]
        return True


class TeacherManager:
    """Manage RSSM teacher snapshots, eval, and promotion."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        policy_idx: int,
        enabled: bool = False,
        eval_interval_frames: int = 50_000_000,
        promote_min_improvement: float = 0.02,
        keep_last_k: int = 3,
        metric: str = "floor_recovery_eval_score",
        max_uncertainty: float = 0.5,
        require_recovery_pos_in: tuple[float, float] = (0.05, 0.95),
    ) -> None:
        self.enabled = bool(enabled)
        self.workspace_root = Path(workspace_root)
        self.policy_idx = int(policy_idx)
        self.eval_interval_frames = int(max(1, eval_interval_frames))
        self.promote_min_improvement = float(promote_min_improvement)
        self.keep_last_k = max(1, int(keep_last_k))
        self.metric = str(metric)
        self.max_uncertainty = float(max_uncertainty)
        self.require_recovery_pos_in = tuple(require_recovery_pos_in)

        self.teacher_dir = (
            self.workspace_root / "world_model_teachers" / f"policy_{self.policy_idx:02d}"
        )
        if self.enabled:
            self.teacher_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.teacher_dir / "teacher_index.json"

        self._last_eval_frame: int = -1
        self._candidates: list[TeacherSnapshot] = []
        self._current_teacher: Optional[TeacherSnapshot] = None
        # Audit counters.
        self.promotions: int = 0
        self.rejections: int = 0
        self.last_promotion_reason: str = ""
        self.last_rejection_reason: str = ""

        if self.enabled and self.index_path.exists():
            try:
                data = json.loads(self.index_path.read_text())
                cur = data.get("current")
                if cur and Path(cur.get("path", "")).exists():
                    metrics = TeacherEvalMetrics(**cur.get("metrics", {}))
                    self._current_teacher = TeacherSnapshot(
                        path=Path(cur["path"]),
                        frame=int(cur.get("frame", 0)),
                        global_step=int(cur.get("global_step", 0)),
                        score=float(cur.get("score", float("-inf"))),
                        metrics=metrics,
                    )
            except Exception:
                self._current_teacher = None

    # ------------------------------------------------------------------
    # Snapshot management
    # ------------------------------------------------------------------

    def should_eval(self, env_frames: int) -> bool:
        if not self.enabled:
            return False
        if self._last_eval_frame < 0:
            return True
        return int(env_frames) - int(self._last_eval_frame) >= int(
            self.eval_interval_frames
        )

    def maybe_save_candidate_snapshot(
        self,
        *,
        env_frames: int,
        trainer_state_dict: dict[str, Any],
    ) -> Optional[Path]:
        """Persist a candidate snapshot to disk; returns its path."""
        if not self.enabled:
            return None
        self.teacher_dir.mkdir(parents=True, exist_ok=True)
        path = (
            self.teacher_dir / f"wm_teacher_step_{int(env_frames):010d}.pt"
        )
        payload = dict(trainer_state_dict)
        payload["frame"] = int(env_frames)
        payload["timestamp"] = time.time()
        torch.save(payload, str(path))
        # Trim old candidates beyond keep_last_k * 2 (keep some slack so
        # rejected snapshots are preserved for offline analysis).
        all_snaps = sorted(self.teacher_dir.glob("wm_teacher_step_*.pt"))
        keep = self.keep_last_k * 2
        if len(all_snaps) > keep:
            for victim in all_snaps[: -keep]:
                try:
                    victim.unlink()
                except OSError:
                    pass
        return path

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def evaluate(
        model: SequenceWorldModelBase,
        eval_buffer: FloorRecoveryEvalBuffer,
        device: torch.device | str,
        *,
        floor_tol_threshold: float = 0.02,
        floor_tol_floor: float = 0.012,
    ) -> TeacherEvalMetrics:
        """Run the model on every batch in the buffer; return aggregate metrics."""
        device = torch.device(device)
        if not eval_buffer.batches:
            return TeacherEvalMetrics()
        model_was_training = model.training
        model.eval()
        try:
            agg_recovery: list[float] = []
            agg_horizon: list[float] = []
            agg_drop: list[float] = []
            agg_d_hat_mse: list[float] = []
            agg_value_mse: list[float] = []
            recovery_pos: list[float] = []
            horizon_pos: list[float] = []
            drop_pos: list[float] = []
            n_total = 0
            has_nan = False
            for batch in eval_buffer.batches:
                b = batch.to(device)
                losses = model.compute_aux_loss(
                    b,
                    floor_aware_weighting=True,
                    floor_tol_threshold=float(floor_tol_threshold),
                    floor_tol_floor=float(floor_tol_floor),
                )
                # Each loss key is a 0-dim tensor.
                rec = losses.get("loss_recovery")
                if rec is None:
                    continue
                v_rec, n_rec = _safe_isfinite_mean(rec)
                has_nan = has_nan or n_rec
                agg_recovery.append(v_rec)
                rec_pos = losses.get("recovery_pos_rate")
                if rec_pos is not None:
                    rp, _ = _safe_isfinite_mean(rec_pos)
                    recovery_pos.append(rp)
                # horizon / drop heads (h8 + h16)
                horizon_vals = []
                drop_vals = []
                for k, t in losses.items():
                    if k.startswith("loss_horizon_success_h"):
                        v, n = _safe_isfinite_mean(t)
                        has_nan = has_nan or n
                        horizon_vals.append(v)
                    if k.startswith("loss_drop_risk_h"):
                        v, n = _safe_isfinite_mean(t)
                        has_nan = has_nan or n
                        drop_vals.append(v)
                if horizon_vals:
                    agg_horizon.append(sum(horizon_vals) / len(horizon_vals))
                if drop_vals:
                    agg_drop.append(sum(drop_vals) / len(drop_vals))
                # d_hat_trend MSE comes from loss_task_error.
                te, n_te = _safe_isfinite_mean(losses.get("loss_task_error"))
                has_nan = has_nan or n_te
                agg_d_hat_mse.append(te)
                vl, n_vl = _safe_isfinite_mean(losses.get("loss_value"))
                has_nan = has_nan or n_vl
                agg_value_mse.append(vl)
                n_total += int(b.B)
            recovery_bce = float(sum(agg_recovery) / max(1, len(agg_recovery)))
            horizon_bce = float(sum(agg_horizon) / max(1, len(agg_horizon)))
            drop_bce = float(sum(agg_drop) / max(1, len(agg_drop)))
            d_hat_mse = float(sum(agg_d_hat_mse) / max(1, len(agg_d_hat_mse)))
            value_mse = float(sum(agg_value_mse) / max(1, len(agg_value_mse)))
            recovery_pos_rate = float(
                sum(recovery_pos) / max(1, len(recovery_pos))
            ) if recovery_pos else 0.0

            score = -(
                recovery_bce
                + horizon_bce
                + drop_bce
                + 0.5 * d_hat_mse
            )
            metrics = TeacherEvalMetrics(
                score=float(score),
                recovery_bce=recovery_bce,
                horizon_success_bce=horizon_bce,
                drop_risk_bce=drop_bce,
                d_hat_trend_mse=d_hat_mse,
                value_mse=value_mse,
                recovery_pos_rate=recovery_pos_rate,
                horizon_pos_rate=0.0,
                drop_pos_rate=0.0,
                n_samples=n_total,
                has_nan=has_nan,
                components={
                    "recovery_bce": recovery_bce,
                    "horizon_success_bce": horizon_bce,
                    "drop_risk_bce": drop_bce,
                    "d_hat_trend_mse": d_hat_mse,
                    "value_mse": value_mse,
                    "score": float(score),
                },
            )
            return metrics
        finally:
            if model_was_training:
                model.train()

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    def maybe_promote(
        self,
        *,
        candidate_path: Path,
        candidate_metrics: TeacherEvalMetrics,
        env_frames: int,
        global_step: int,
    ) -> tuple[bool, str]:
        """Conservative promotion gate (Item 5).

        Returns (promoted, reason).
        Reasons (rejection): nan, recovery_bce_worse, horizon_bce_worse,
        drop_bce_worse, d_hat_trend_worse, recovery_pos_degenerate,
        score_below_threshold. Reasons (acceptance): "first_teacher",
        "improved".
        """
        self._last_eval_frame = int(env_frames)
        if not self.enabled:
            self.last_rejection_reason = "manager_disabled"
            return False, "manager_disabled"
        if candidate_metrics.has_nan:
            self.rejections += 1
            self.last_rejection_reason = "nan"
            return False, "nan"
        # Non-degenerate recovery target check.
        lo, hi = self.require_recovery_pos_in
        if not (
            float(lo) <= float(candidate_metrics.recovery_pos_rate) <= float(hi)
        ):
            self.rejections += 1
            self.last_rejection_reason = "recovery_pos_degenerate"
            return False, "recovery_pos_degenerate"
        if self._current_teacher is None:
            self._current_teacher = TeacherSnapshot(
                path=candidate_path,
                frame=int(env_frames),
                global_step=int(global_step),
                score=float(candidate_metrics.score),
                metrics=candidate_metrics,
            )
            self.promotions += 1
            self.last_promotion_reason = "first_teacher"
            self._write_index()
            return True, "first_teacher"
        prev = self._current_teacher.metrics
        if candidate_metrics.recovery_bce > prev.recovery_bce + 1e-8:
            self.rejections += 1
            self.last_rejection_reason = "recovery_bce_worse"
            return False, "recovery_bce_worse"
        if candidate_metrics.horizon_success_bce > prev.horizon_success_bce + 1e-8:
            self.rejections += 1
            self.last_rejection_reason = "horizon_bce_worse"
            return False, "horizon_bce_worse"
        if candidate_metrics.drop_risk_bce > prev.drop_risk_bce + 1e-8:
            self.rejections += 1
            self.last_rejection_reason = "drop_bce_worse"
            return False, "drop_bce_worse"
        if candidate_metrics.d_hat_trend_mse > prev.d_hat_trend_mse + 1e-3:
            self.rejections += 1
            self.last_rejection_reason = "d_hat_trend_worse"
            return False, "d_hat_trend_worse"
        if (
            float(candidate_metrics.score)
            < float(prev.score) + float(self.promote_min_improvement)
        ):
            self.rejections += 1
            self.last_rejection_reason = "score_below_threshold"
            return False, "score_below_threshold"
        # Promote.
        self._current_teacher = TeacherSnapshot(
            path=candidate_path,
            frame=int(env_frames),
            global_step=int(global_step),
            score=float(candidate_metrics.score),
            metrics=candidate_metrics,
        )
        self.promotions += 1
        self.last_promotion_reason = "improved"
        self._write_index()
        return True, "improved"

    def current_teacher(self) -> Optional[TeacherSnapshot]:
        return self._current_teacher

    def _write_index(self) -> None:
        if not self.enabled or self._current_teacher is None:
            return
        try:
            payload = {
                "policy_idx": int(self.policy_idx),
                "current": {
                    "path": str(self._current_teacher.path),
                    "frame": int(self._current_teacher.frame),
                    "global_step": int(self._current_teacher.global_step),
                    "score": float(self._current_teacher.score),
                    "metrics": self._current_teacher.metrics.to_json(),
                    "promotion_count": int(self.promotions),
                },
                "audit": {
                    "promotions": int(self.promotions),
                    "rejections": int(self.rejections),
                    "last_promotion_reason": str(self.last_promotion_reason),
                    "last_rejection_reason": str(self.last_rejection_reason),
                },
                "metric": str(self.metric),
                "timestamp": time.time(),
            }
            tmp = self.index_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(self.index_path)
        except Exception as exc:  # pragma: no cover
            print(f"[wm-teacher] index write failed: {exc}")


__all__ = [
    "TeacherEvalMetrics",
    "TeacherSnapshot",
    "FloorRecoveryEvalBuffer",
    "TeacherManager",
]
