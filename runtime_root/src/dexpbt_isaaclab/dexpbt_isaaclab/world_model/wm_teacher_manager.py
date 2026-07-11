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
    """Per-snapshot evaluation metrics on the held-out floor buffer.

    Phase A4 P5 (2026-06-10): added per-horizon pos_rate fields and
    ``uncertainty`` so the promotion gate can reason about target
    non-degeneracy at the head level instead of relying on recovery only.
    Backward compat: legacy ``horizon_pos_rate`` / ``drop_pos_rate``
    aggregate scalars are still populated by ``evaluate``.
    """

    score: float = 0.0
    recovery_bce: float = 0.0
    horizon_success_bce: float = 0.0
    drop_risk_bce: float = 0.0
    d_hat_trend_mse: float = 0.0
    value_mse: float = 0.0
    recovery_pos_rate: float = 0.0
    horizon_pos_rate: float = 0.0
    drop_pos_rate: float = 0.0
    horizon_success_pos_rate_h4: float = 0.0
    horizon_success_pos_rate_h8: float = 0.0
    drop_risk_pos_rate_h4: float = 0.0
    drop_risk_pos_rate_h8: float = 0.0
    uncertainty: float = 0.0
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
        keep_last_k: int = 6,
        metric: str = "floor_recovery_eval_score",
        max_uncertainty: float = 0.5,
        require_recovery_pos_in: tuple[float, float] = (0.05, 0.95),
        # Phase A4 P5 (2026-06-10): teacher-gate fix knobs.
        allow_first_promotion: bool = True,
        min_eval_size: int = 512,
        pos_rate_low: float = 0.001,
        pos_rate_high: float = 0.999,
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
        # Phase A4 P5 (2026-06-10): teacher-gate fix knobs.
        self.allow_first_promotion = bool(allow_first_promotion)
        self.min_eval_size = max(0, int(min_eval_size))
        self.pos_rate_low = float(pos_rate_low)
        self.pos_rate_high = float(pos_rate_high)

        self.teacher_dir = (
            self.workspace_root / "world_model_teachers" / f"policy_{self.policy_idx:02d}"
        )
        if self.enabled:
            self.teacher_dir.mkdir(parents=True, exist_ok=True)
        # Phase A4 P5 (2026-06-10): canonical name is ``index.json``.
        # ``teacher_index.json`` is read once for backward compat.
        self.index_path = self.teacher_dir / "index.json"
        self.events_path = self.teacher_dir / "teacher_eval_events.jsonl"
        self._legacy_index_path = self.teacher_dir / "teacher_index.json"

        self._last_eval_frame: int = -1
        self._candidates: list[TeacherSnapshot] = []
        self._current_teacher: Optional[TeacherSnapshot] = None
        # Audit counters.
        self.promotions: int = 0
        self.rejections: int = 0
        self.eval_attempts: int = 0
        self.last_promotion_reason: str = ""
        self.last_rejection_reason: str = ""
        self.last_eval_score: float = 0.0
        self.last_improvement: float = 0.0
        self.last_target_degenerate: bool = False
        self.last_no_eval_data: bool = False
        self.last_eval_buffer_size: int = 0

        for path in (self.index_path, self._legacy_index_path):
            if not (self.enabled and path.exists()):
                continue
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            cur = data.get("current")
            if cur and Path(cur.get("path", "")).exists():
                # Filter to TeacherEvalMetrics-known fields so legacy
                # payloads with extra components do not crash __init__.
                allowed = set(TeacherEvalMetrics().__dict__.keys())
                metrics = TeacherEvalMetrics(
                    **{k: v for k, v in (cur.get("metrics") or {}).items()
                       if k in allowed}
                )
                self._current_teacher = TeacherSnapshot(
                    path=Path(cur["path"]),
                    frame=int(cur.get("frame", 0)),
                    global_step=int(cur.get("global_step", 0)),
                    score=float(cur.get("score", float("-inf"))),
                    metrics=metrics,
                )
            audit = data.get("audit") or {}
            self.promotions = int(audit.get("promotions", self.promotions))
            self.rejections = int(audit.get("rejections", self.rejections))
            self.eval_attempts = int(
                audit.get("eval_attempts", self.eval_attempts)
            )
            break

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
        # Phase A4 P5 (2026-06-10): keep exactly ``keep_last_k`` snapshots
        # (no ×2 slack). Default 6 means six latest candidates retained
        # for offline forensics — including rejected ones.
        all_snaps = sorted(self.teacher_dir.glob("wm_teacher_step_*.pt"))
        keep = max(1, int(self.keep_last_k))
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
        """Run the model on every batch in the buffer; return aggregate metrics.

        Phase A4 P5 (2026-06-10): also captures per-horizon pos_rate from
        ``compute_aux_loss``'s float-metric companion (when present) so the
        promotion gate can reason about head-level non-degeneracy.
        """
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
            # Per-horizon pos_rate accumulators. Keys are integer h_steps
            # (typically 4, 8). Only populated when the model's
            # compute_aux_loss returns the float-metric dict variant.
            horizon_success_pos: dict[int, list[float]] = {}
            drop_risk_pos: dict[int, list[float]] = {}
            n_total = 0
            has_nan = False
            for batch in eval_buffer.batches:
                b = batch.to(device)
                # Some models return a float-metric dict separately when
                # ``return_metrics`` is requested; otherwise compute_aux_loss
                # returns a tensor dict only. Try the metrics path first.
                metrics_floats: dict[str, float] = {}
                try:
                    losses = model.compute_aux_loss(
                        b,
                        floor_aware_weighting=True,
                        floor_tol_threshold=float(floor_tol_threshold),
                        floor_tol_floor=float(floor_tol_floor),
                        return_metrics=True,
                    )
                    if isinstance(losses, tuple) and len(losses) == 2:
                        losses, metrics_floats = losses  # type: ignore[assignment]
                except TypeError:
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
                # horizon / drop heads (h4 + h8 by default)
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
                # Capture per-horizon pos_rate from the float-metric dict.
                for mk, mv in metrics_floats.items():
                    if mk.startswith("horizon_success_pos_rate_h"):
                        try:
                            h = int(mk.replace("horizon_success_pos_rate_h", ""))
                        except ValueError:
                            continue
                        horizon_success_pos.setdefault(h, []).append(float(mv))
                    elif mk.startswith("drop_risk_pos_rate_h"):
                        try:
                            h = int(mk.replace("drop_risk_pos_rate_h", ""))
                        except ValueError:
                            continue
                        drop_risk_pos.setdefault(h, []).append(float(mv))
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

            def _avg(d: dict[int, list[float]], key: int) -> float:
                vals = d.get(key, [])
                return float(sum(vals) / max(1, len(vals))) if vals else 0.0

            hs_h4 = _avg(horizon_success_pos, 4)
            hs_h8 = _avg(horizon_success_pos, 8)
            dr_h4 = _avg(drop_risk_pos, 4)
            dr_h8 = _avg(drop_risk_pos, 8)
            # Aggregate horizon/drop pos rate (legacy field).
            all_hs = [v for vs in horizon_success_pos.values() for v in vs]
            all_dr = [v for vs in drop_risk_pos.values() for v in vs]
            horizon_pos_rate_legacy = (
                float(sum(all_hs) / len(all_hs)) if all_hs else 0.0
            )
            drop_pos_rate_legacy = (
                float(sum(all_dr) / len(all_dr)) if all_dr else 0.0
            )

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
                horizon_pos_rate=horizon_pos_rate_legacy,
                drop_pos_rate=drop_pos_rate_legacy,
                horizon_success_pos_rate_h4=hs_h4,
                horizon_success_pos_rate_h8=hs_h8,
                drop_risk_pos_rate_h4=dr_h4,
                drop_risk_pos_rate_h8=dr_h8,
                uncertainty=0.0,
                n_samples=n_total,
                has_nan=has_nan,
                components={
                    "recovery_bce": recovery_bce,
                    "horizon_success_bce": horizon_bce,
                    "drop_risk_bce": drop_bce,
                    "d_hat_trend_mse": d_hat_mse,
                    "value_mse": value_mse,
                    "score": float(score),
                    "horizon_success_pos_rate_h4": hs_h4,
                    "horizon_success_pos_rate_h8": hs_h8,
                    "drop_risk_pos_rate_h4": dr_h4,
                    "drop_risk_pos_rate_h8": dr_h8,
                },
            )
            return metrics
        finally:
            if model_was_training:
                model.train()

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Helpers — head-level non-degeneracy
    # ------------------------------------------------------------------

    def _is_pos_rate_degenerate(self, value: float) -> bool:
        """Pos-rate is degenerate when outside ``(low, high)`` (inclusive).

        Default range (0.001, 0.999) only rejects truly-saturated targets;
        anything strictly between is considered to carry signal.
        """
        v = float(value)
        return v <= float(self.pos_rate_low) or v >= float(self.pos_rate_high)

    def _classify_target_degeneracy(
        self, m: TeacherEvalMetrics
    ) -> tuple[bool, dict[str, bool]]:
        """Return (fully_degenerate, per-head dict).

        Phase A4 P5 (2026-06-10): a target is *fully* degenerate when ALL
        of the task heads (horizon-success h4 & h8) are saturated. The
        recovery head pos rate is reported but does not by itself force
        a rejection (drop is naturally rare; recovery is naturally weak
        before hard-floor). This is the primary non-degenerate check.
        """
        per_head = {
            "horizon_success_h4": self._is_pos_rate_degenerate(
                m.horizon_success_pos_rate_h4
            ),
            "horizon_success_h8": self._is_pos_rate_degenerate(
                m.horizon_success_pos_rate_h8
            ),
            "drop_risk_h4": self._is_pos_rate_degenerate(
                m.drop_risk_pos_rate_h4
            ),
            "drop_risk_h8": self._is_pos_rate_degenerate(
                m.drop_risk_pos_rate_h8
            ),
            "recovery": self._is_pos_rate_degenerate(m.recovery_pos_rate),
        }
        # Fully degenerate when BOTH horizon-success heads are saturated.
        # Drop_risk is excluded from the hard check because drop is rare.
        fully = bool(
            per_head["horizon_success_h4"] and per_head["horizon_success_h8"]
        )
        return fully, per_head

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
        """Conservative promotion gate.

        Phase A4 P5 (2026-06-10) gate logic:
          1. ``manager_disabled`` short circuit.
          2. ``nan`` reject.
          3. ``no_eval_data`` reject when ``n_samples < min_eval_size``
             (typically because the eval buffer was empty).
          4. ``high_uncertainty`` reject when ``uncertainty > max_uncertainty``.
          5. **Primary** non-degenerate check uses horizon-success pos
             rates (h4 & h8). If BOTH are saturated → ``target_fully_degenerate``.
             ``recovery_pos_rate`` saturation only emits a warning and is
             *not* a hard reject for the first teacher.
          6. First-teacher path:
             * If ``allow_first_promotion=True`` → promote with reason
               ``first_teacher``.
             * If ``allow_first_promotion=False`` → reject with reason
               ``first_promotion_disabled``.
          7. Second+ teacher: keep the conservative non-degradation gate
             (recovery_bce, horizon_bce, drop_bce, d_hat_trend_mse) plus
             ``promote_min_improvement`` on the score. Recovery_bce/etc.
             non-degradation tolerance is unchanged.

        Persistence side-effect: every call appends to
        ``teacher_eval_events.jsonl`` and rewrites ``index.json`` with the
        current audit state, regardless of whether promotion happened.
        """
        self._last_eval_frame = int(env_frames)
        self.eval_attempts += 1
        self.last_eval_score = float(candidate_metrics.score)
        self.last_no_eval_data = False
        prev_score = (
            float(self._current_teacher.score)
            if self._current_teacher is not None else float("nan")
        )
        improvement = (
            float(candidate_metrics.score) - prev_score
            if self._current_teacher is not None else 0.0
        )
        self.last_improvement = improvement

        def _finalize(promoted: bool, reason: str) -> tuple[bool, str]:
            if promoted:
                self.promotions += 1
                self.last_promotion_reason = reason
            else:
                self.rejections += 1
                self.last_rejection_reason = reason
            self._write_index()
            self._append_event(
                env_frames=env_frames,
                candidate_path=candidate_path,
                metrics=candidate_metrics,
                promoted=promoted,
                reason=reason,
                improvement=improvement,
                prev_score=prev_score,
            )
            return promoted, reason

        if not self.enabled:
            self.last_rejection_reason = "manager_disabled"
            return False, "manager_disabled"

        # 2) NaN rejection.
        if candidate_metrics.has_nan:
            return _finalize(False, "nan")

        # 3) Min-eval-size gate (no_eval_data).
        self.last_eval_buffer_size = int(candidate_metrics.n_samples)
        if int(candidate_metrics.n_samples) < int(self.min_eval_size):
            self.last_no_eval_data = True
            return _finalize(False, "no_eval_data")

        # 4) Uncertainty gate.
        if float(candidate_metrics.uncertainty) > float(self.max_uncertainty):
            return _finalize(False, "high_uncertainty")

        # 5) Target non-degeneracy via horizon-success pos rates.
        fully_degenerate, _per_head = self._classify_target_degeneracy(
            candidate_metrics
        )
        self.last_target_degenerate = bool(fully_degenerate)
        if fully_degenerate:
            return _finalize(False, "target_fully_degenerate")

        # 6) First-teacher path.
        if self._current_teacher is None:
            if not self.allow_first_promotion:
                return _finalize(False, "first_promotion_disabled")
            self._current_teacher = TeacherSnapshot(
                path=candidate_path,
                frame=int(env_frames),
                global_step=int(global_step),
                score=float(candidate_metrics.score),
                metrics=candidate_metrics,
            )
            return _finalize(True, "first_teacher")

        # 7) Conservative non-degradation gate for second+ teacher.
        prev = self._current_teacher.metrics
        if candidate_metrics.recovery_bce > prev.recovery_bce + 1e-8:
            return _finalize(False, "recovery_bce_worse")
        if candidate_metrics.horizon_success_bce > prev.horizon_success_bce + 1e-8:
            return _finalize(False, "horizon_bce_worse")
        if candidate_metrics.drop_risk_bce > prev.drop_risk_bce + 1e-8:
            return _finalize(False, "drop_bce_worse")
        if candidate_metrics.d_hat_trend_mse > prev.d_hat_trend_mse + 1e-3:
            return _finalize(False, "d_hat_trend_worse")
        if (
            float(candidate_metrics.score)
            < float(prev.score) + float(self.promote_min_improvement)
        ):
            return _finalize(False, "score_below_threshold")
        # Promote.
        self._current_teacher = TeacherSnapshot(
            path=candidate_path,
            frame=int(env_frames),
            global_step=int(global_step),
            score=float(candidate_metrics.score),
            metrics=candidate_metrics,
        )
        return _finalize(True, "improved")

    # ------------------------------------------------------------------
    # No-eval-data path (called from train.py when the buffer is empty)
    # ------------------------------------------------------------------

    def record_no_eval_data(
        self,
        *,
        env_frames: int,
        eval_buffer_size: int,
    ) -> tuple[bool, str]:
        """Persist a 'no_eval_data' attempt without invoking the model.

        ``train.py`` calls this when ``len(eval_buffer)==0`` so we still
        increment the attempt counter, write index.json, and append an
        events row — the user requirement (A.1, A.2) for *every* attempt.
        """
        self._last_eval_frame = int(env_frames)
        self.eval_attempts += 1
        self.last_eval_score = 0.0
        self.last_improvement = 0.0
        self.last_target_degenerate = False
        self.last_no_eval_data = True
        self.last_eval_buffer_size = int(max(0, eval_buffer_size))
        if not self.enabled:
            self.last_rejection_reason = "manager_disabled"
            return False, "manager_disabled"
        self.rejections += 1
        self.last_rejection_reason = "no_eval_data"
        self._write_index()
        self._append_event(
            env_frames=env_frames,
            candidate_path=None,
            metrics=TeacherEvalMetrics(),
            promoted=False,
            reason="no_eval_data",
            improvement=0.0,
            prev_score=float("nan"),
        )
        return False, "no_eval_data"

    def current_teacher(self) -> Optional[TeacherSnapshot]:
        return self._current_teacher

    def _write_index(self) -> None:
        """Phase A4 P5 (2026-06-10): write index.json on EVERY eval attempt.

        Unlike the legacy implementation that only wrote on promotion, we
        now persist eval state regardless of outcome so the analyzer can
        reconstruct rejection history. ``current`` may be null when no
        teacher has ever been promoted.
        """
        if not self.enabled:
            return
        try:
            current_payload: Optional[dict[str, Any]] = None
            if self._current_teacher is not None:
                current_payload = {
                    "path": str(self._current_teacher.path),
                    "frame": int(self._current_teacher.frame),
                    "global_step": int(self._current_teacher.global_step),
                    "score": float(self._current_teacher.score),
                    "metrics": self._current_teacher.metrics.to_json(),
                }
            cur_metrics = (
                self._current_teacher.metrics
                if self._current_teacher is not None
                else None
            )
            payload = {
                "policy_idx": int(self.policy_idx),
                "current": current_payload,
                "current_teacher_step": int(
                    self._current_teacher.frame
                    if self._current_teacher is not None else -1
                ),
                "current_teacher_path": str(
                    self._current_teacher.path
                    if self._current_teacher is not None else ""
                ),
                "is_teacher_valid": bool(self._current_teacher is not None),
                "audit": {
                    "promotions": int(self.promotions),
                    "rejections": int(self.rejections),
                    "eval_attempts": int(self.eval_attempts),
                    "last_promotion_reason": str(self.last_promotion_reason),
                    "last_rejection_reason": str(self.last_rejection_reason),
                },
                # Top-level fields requested by the user spec (A.1).
                "eval_attempt_count": int(self.eval_attempts),
                "promotion_count": int(self.promotions),
                "rejection_count": int(self.rejections),
                "last_eval_frame": int(self._last_eval_frame),
                "last_eval_score": float(self.last_eval_score),
                "last_rejected_reason": str(self.last_rejection_reason),
                "last_promoted_reason": str(self.last_promotion_reason),
                "eval_buffer_size": int(self.last_eval_buffer_size),
                "recovery_pos_rate": (
                    float(cur_metrics.recovery_pos_rate)
                    if cur_metrics is not None else 0.0
                ),
                "horizon_success_pos_rate_h4": (
                    float(cur_metrics.horizon_success_pos_rate_h4)
                    if cur_metrics is not None else 0.0
                ),
                "horizon_success_pos_rate_h8": (
                    float(cur_metrics.horizon_success_pos_rate_h8)
                    if cur_metrics is not None else 0.0
                ),
                "drop_risk_pos_rate_h4": (
                    float(cur_metrics.drop_risk_pos_rate_h4)
                    if cur_metrics is not None else 0.0
                ),
                "drop_risk_pos_rate_h8": (
                    float(cur_metrics.drop_risk_pos_rate_h8)
                    if cur_metrics is not None else 0.0
                ),
                "uncertainty": (
                    float(cur_metrics.uncertainty)
                    if cur_metrics is not None else 0.0
                ),
                "target_degenerate": bool(self.last_target_degenerate),
                "no_eval_data": bool(self.last_no_eval_data),
                "metric": str(self.metric),
                "timestamp": time.time(),
            }
            tmp = self.index_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(self.index_path)
        except Exception as exc:  # pragma: no cover
            print(f"[wm-teacher] index write failed: {exc}")

    def _append_event(
        self,
        *,
        env_frames: int,
        candidate_path: Optional[Path],
        metrics: TeacherEvalMetrics,
        promoted: bool,
        reason: str,
        improvement: float,
        prev_score: float,
    ) -> None:
        """Append a single line to teacher_eval_events.jsonl (A.2)."""
        if not self.enabled:
            return
        try:
            row = {
                "ts": time.time(),
                "frame": int(env_frames),
                "candidate_path": str(candidate_path) if candidate_path else "",
                "eval_score": float(metrics.score),
                "promoted": bool(promoted),
                "rejected_reason": (
                    str(reason) if not promoted else ""
                ),
                "promoted_reason": str(reason) if promoted else "",
                "eval_buffer_size": int(self.last_eval_buffer_size),
                "n_samples": int(metrics.n_samples),
                "recovery_pos_rate": float(metrics.recovery_pos_rate),
                "horizon_success_pos_rate_h4": float(
                    metrics.horizon_success_pos_rate_h4
                ),
                "horizon_success_pos_rate_h8": float(
                    metrics.horizon_success_pos_rate_h8
                ),
                "drop_risk_pos_rate_h4": float(metrics.drop_risk_pos_rate_h4),
                "drop_risk_pos_rate_h8": float(metrics.drop_risk_pos_rate_h8),
                "recovery_bce": float(metrics.recovery_bce),
                "horizon_success_bce": float(metrics.horizon_success_bce),
                "drop_risk_bce": float(metrics.drop_risk_bce),
                "d_hat_trend_mse": float(metrics.d_hat_trend_mse),
                "value_mse": float(metrics.value_mse),
                "uncertainty": float(metrics.uncertainty),
                "previous_teacher_score": (
                    None if (prev_score != prev_score) else float(prev_score)
                ),
                "improvement": float(improvement),
                "has_nan": bool(metrics.has_nan),
                "target_degenerate": bool(self.last_target_degenerate),
                "no_eval_data": bool(self.last_no_eval_data),
            }
            self.teacher_dir.mkdir(parents=True, exist_ok=True)
            with open(self.events_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        except Exception as exc:  # pragma: no cover
            print(f"[wm-teacher] event append failed: {exc}")


__all__ = [
    "TeacherEvalMetrics",
    "TeacherSnapshot",
    "FloorRecoveryEvalBuffer",
    "TeacherManager",
]
