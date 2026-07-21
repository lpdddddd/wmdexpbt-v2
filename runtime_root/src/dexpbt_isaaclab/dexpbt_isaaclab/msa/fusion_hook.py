"""Main-training MSA integration hook.

Triggers / observes existing MSA cycles without replacing MSAController
eval or state-machine infrastructure. Sidecar remains the production
cycle driver; this hook:

- maintains per-worker WorkerMSAState on the curriculum control chain;
- gates BUILD readiness / HOLD cooldown;
- under fusion profile, turns mastery shrink into proposals only;
- applies ADVANCE / HOLD / ROLLBACK control-state updates after a cycle
  decision artifact appears;
- exposes an optional in-process test hook (disabled on formal profiles).

Difference vs pure pilot sidecar continuation: fused main training may
run without a continuation_guard first-tick barrier; the sidecar still
owns paired eval + durable tolerance commit when present. Observer applies
worker_state mode transitions from decision artifacts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

from dexpbt_isaaclab.msa.decision import (
    SeedEvalResult,
    decide_selective_promotion,
    l_mid,
    same_tau_retention,
)
from dexpbt_isaaclab.msa.fusion_config import MSAFusionConfig, print_effective_config
from dexpbt_isaaclab.msa.fusion_events import emit_event, write_cycle_archive
from dexpbt_isaaclab.msa.worker_state import (
    MODE_BUILD,
    MODE_HOLD,
    MODE_PROBE,
    WorkerMSAState,
    apply_advance,
    apply_rollback_hold,
    enter_probe,
    load_worker_state,
    maybe_exit_hold,
    migrate_worker_state,
    save_worker_state,
)

BUSY_FILE = "MSA_BUSY.json"
DECISION_FILE = "MSA_FUSION_DECISION.json"
WORKER_STATE_DIR = "msa_worker_state"


def busy_path(run_dir: Path) -> Path:
    return Path(run_dir) / BUSY_FILE


def decision_path(run_dir: Path) -> Path:
    return Path(run_dir) / DECISION_FILE


def worker_state_path(run_dir: Path, worker_id: int) -> Path:
    return Path(run_dir) / WORKER_STATE_DIR / f"{int(worker_id):03d}.json"


def read_busy(run_dir: Path) -> Optional[dict[str, Any]]:
    p = busy_path(run_dir)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def is_msa_busy(run_dir: Path) -> bool:
    return read_busy(run_dir) is not None


class MSAFusionHook:
    """Per-run fusion controller facade used by the observer / tests."""

    def __init__(
        self,
        run_dir: Path | str,
        cfg: MSAFusionConfig,
        *,
        num_workers: int = 6,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.cfg = cfg
        self.num_workers = int(num_workers)
        self._log = log or (lambda m: print(m, flush=True))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / WORKER_STATE_DIR).mkdir(parents=True, exist_ok=True)
        if cfg.msa_integrated_enabled:
            print_effective_config(cfg, sink=self._log)

    def load_or_migrate(
        self,
        worker_id: int,
        *,
        current_tolerance: float,
        anchor_checkpoint: str = "",
    ) -> WorkerMSAState:
        path = worker_state_path(self.run_dir, worker_id)
        raw = None
        if path.is_file():
            try:
                raw = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                raw = None
        st, logs = migrate_worker_state(
            raw,
            worker_id=worker_id,
            current_tolerance=current_tolerance,
            anchor_checkpoint=anchor_checkpoint,
            cfg=self.cfg,
        )
        for line in logs:
            emit_event(self.run_dir, "MSA_STATE_RESTORED",
                       {"worker_id": worker_id, "message": line})
            self._log(f"[msa-fusion] {line}")
        save_worker_state(path, st)
        return st

    def save(self, state: WorkerMSAState) -> None:
        save_worker_state(worker_state_path(self.run_dir, state.worker_id), state)

    def tick_hold_exit(self, state: WorkerMSAState, frame: int) -> WorkerMSAState:
        before = state.mode
        state = maybe_exit_hold(state, frame)
        if before == MODE_HOLD and state.mode == MODE_BUILD:
            emit_event(self.run_dir, "MSA_TRIGGER",
                       {"worker_id": state.worker_id, "reason": "hold_exit",
                        "frame": frame})
        return state

    def readiness_allows_probe(self, state: WorkerMSAState, frame: int) -> bool:
        if not self.cfg.msa_integrated_enabled:
            return False
        if state.mode == MODE_HOLD and frame < state.hold_until_frame:
            return False
        if state.mode == MODE_PROBE or state.transaction_open:
            return False
        if state.consecutive_failures >= self.cfg.msa_max_consecutive_failures:
            return False
        stage = self.cfg.stage_for_tolerance(state.current_tolerance)
        if stage == "BUILD":
            return False  # BUILD: no paired probe; curriculum via proposals only
        if (state.last_decision_frame >= 0
                and frame - state.last_decision_frame
                < self.cfg.msa_cycle_interval_frames):
            return False
        return True

    def start_probe(
        self,
        state: WorkerMSAState,
        *,
        frame: int,
        cycle_id: str | None = None,
    ) -> WorkerMSAState:
        target = self.cfg.probe_target(state.current_tolerance)
        cid = cycle_id or f"fusion_{int(time.time())}_{state.worker_id}"
        state = enter_probe(
            state,
            cycle_id=cid,
            target_tol=target,
            frame=frame,
            budget_frames=self.cfg.msa_probe_budget_frames,
        )
        emit_event(self.run_dir, "MSA_PROBE_START", {
            "worker_id": state.worker_id,
            "cycle_id": cid,
            "tau": state.anchor_tolerance,
            "tau_plus": target,
            "frame": frame,
            "budget": self.cfg.msa_probe_budget_frames,
            "stage": self.cfg.stage_for_tolerance(state.current_tolerance),
        })
        self.save(state)
        return state

    def apply_decision_from_metrics(
        self,
        state: WorkerMSAState,
        *,
        frame: int,
        l_lcb: float,
        rho: float,
        rho_available: bool,
        promote: bool,
        reason: str,
        metrics: dict[str, Any] | None = None,
        anchor_checkpoint: str = "",
    ) -> WorkerMSAState:
        """Apply advance or rollback/hold using pilot-aligned gates.

        Advance iff promote and (L_LCB > 0 and rho available and
        rho >= msa_retention_min). WM signals must NOT be passed here as
        sole triggers — caller must use real eval metrics.
        """
        metrics = dict(metrics or {})
        metrics.update({"l_lcb": l_lcb, "rho": rho, "rho_available": rho_available})
        ok = (
            bool(promote)
            and float(l_lcb) > 0.0
            and bool(rho_available)
            and float(rho) >= float(self.cfg.msa_retention_min)
        )
        if ok:
            new_tol = float(state.pending_target_tolerance or
                            self.cfg.probe_target(state.current_tolerance))
            ckpt = anchor_checkpoint or state.anchor_checkpoint
            state = apply_advance(
                state,
                new_tol=new_tol,
                frame=frame,
                anchor_checkpoint=ckpt,
                metrics=metrics,
                reason=reason or "advance",
                cfg=self.cfg,
            )
            emit_event(self.run_dir, "MSA_ADVANCE_COMMIT", {
                "worker_id": state.worker_id,
                "new_tol": new_tol,
                "frame": frame,
                "reason": state.decision_reason,
            })
        else:
            state = apply_rollback_hold(
                state,
                frame=frame,
                cooldown_frames=self.cfg.msa_hold_cooldown_frames,
                metrics=metrics,
                reason=reason or "rollback_hold",
            )
            emit_event(self.run_dir, "MSA_ROLLBACK", {
                "worker_id": state.worker_id,
                "frame": frame,
                "reason": state.decision_reason,
                "hold_until": state.hold_until_frame,
            })
            emit_event(self.run_dir, "MSA_HOLD", {
                "worker_id": state.worker_id,
                "hold_until": state.hold_until_frame,
            })
        self.save(state)
        return state

    def consume_sidecar_decision_file(self) -> Optional[dict[str, Any]]:
        """Read and clear MSA_FUSION_DECISION.json if present."""
        p = decision_path(self.run_dir)
        if not p.is_file():
            return None
        try:
            doc = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            emit_event(self.run_dir, "MSA_FAIL_CLOSED",
                       {"error": "unreadable_decision_file"})
            return None
        try:
            p.unlink()
        except OSError:
            pass
        return doc

    def apply_sidecar_decision_doc(
        self,
        doc: dict[str, Any],
        states: dict[int, WorkerMSAState],
        *,
        frame_by_worker: dict[int, int] | None = None,
    ) -> dict[int, WorkerMSAState]:
        """Map a sidecar / injector decision document onto worker states."""
        frame_by_worker = frame_by_worker or {}
        decisions = doc.get("decisions") or {}
        for w_str, d in decisions.items():
            w = int(w_str)
            if w not in states:
                continue
            st = states[w]
            frame = int(frame_by_worker.get(w, st.last_decision_frame
                                            if st.last_decision_frame >= 0
                                            else 0))
            action = str(d.get("action", "hold")).lower()
            metrics = dict(d.get("metrics") or {})
            if action == "advance":
                st = self.apply_decision_from_metrics(
                    st,
                    frame=frame,
                    l_lcb=float(d.get("l_lcb", metrics.get("l_lcb", 1.0))),
                    rho=float(d.get("rho", metrics.get("rho", 1.0))),
                    rho_available=bool(d.get("rho_available", True)),
                    promote=True,
                    reason=str(d.get("reason", "sidecar_advance")),
                    metrics=metrics,
                    anchor_checkpoint=str(d.get("anchor_checkpoint",
                                                st.anchor_checkpoint)),
                )
            else:
                st = self.apply_decision_from_metrics(
                    st,
                    frame=frame,
                    l_lcb=float(d.get("l_lcb", metrics.get("l_lcb", -1.0))),
                    rho=float(d.get("rho", metrics.get("rho", 0.0))),
                    rho_available=bool(d.get("rho_available", True)),
                    promote=False,
                    reason=str(d.get("reason", "sidecar_hold_or_rollback")),
                    metrics=metrics,
                )
            states[w] = st
        write_cycle_archive(
            self.run_dir,
            str(doc.get("intervention_id") or f"decision_{int(time.time())}"),
            doc,
        )
        return states

    def tensorboard_scalars(self, state: WorkerMSAState) -> dict[str, float]:
        mode_code = {"BUILD": 0.0, "PROBE": 1.0, "HOLD": 2.0}.get(state.mode, -1.0)
        return {
            "msa/current_tolerance": float(state.current_tolerance),
            "msa/anchor_tolerance": float(state.anchor_tolerance),
            "msa/mode": mode_code,
            "msa/l_mid": float(state.decision_metrics.get("l_mid", 0.0) or 0.0),
            "msa/l_lcb": float(state.decision_metrics.get("l_lcb", 0.0) or 0.0),
            "msa/retention": float(state.decision_metrics.get("rho", 0.0) or 0.0),
            "msa/retention_lcb": float(
                state.decision_metrics.get("rho_lcb", 0.0) or 0.0),
            "msa/consecutive_failures": float(state.consecutive_failures),
            "msa/advance_count": float(state.advance_count),
            "msa/rollback_count": float(state.rollback_count),
            "msa/hold_count": float(state.hold_count),
            "msa/probe_frames": float(state.probe_budget_frames),
        }


def wm_signals_must_not_advance(wm_payload: dict[str, Any]) -> bool:
    """Guard: WM-only payloads cannot satisfy advance. Returns True if blocked."""
    # If only WM keys are present without real eval keys, block.
    real_keys = {"l_lcb", "rho", "l_mid", "p_pre", "p_post", "success"}
    wm_keys = {
        "wm_recovery_prob", "predicted_success", "predicted_drop",
        "snippet_priority", "actor_distill_loss", "wm_imagined_recovery_prob",
    }
    keys = set(wm_payload.keys())
    if keys and keys <= wm_keys:
        return True
    if not (keys & real_keys) and (keys & wm_keys):
        return True
    return False
