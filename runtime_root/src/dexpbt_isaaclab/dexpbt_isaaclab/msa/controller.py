"""MSA probe controller: drives the state machine over the existing
curriculum control-state chain.

Integration contract (no trainer surgery):
- The env already obeys the control JSON written by
  pbt_curriculum_p1.write_curriculum_control: `forced_success_tolerance`
  overrides the live tolerance, and the msa_* fields ride in the same JSON
  (and therefore in the same resume snapshot chain).
- The controller therefore runs as a SIDECAR process next to the training
  supervisor (launched by the MSA arm launcher): it reads per-worker tick
  metas (frame ledger) and control JSONs, transitions the shared state
  machine, and writes:
    * forced_success_tolerance = tau_plus  while a probe segment is active,
    * forced_success_tolerance = tau       when the segment ends,
    * msa_provenance / msa_state_machine   after every transition.
- It NEVER reloads checkpoints, never touches policy/optimizer/replay/PBT
  lineage, and never resets curriculum state: the only mutation channel is
  the tolerance override that the base curriculum already honors.
- Probe frames advance on the worker's own monotonic frame ledger (tick
  meta `frame`), so every probe frame is charged against the 100M
  continuation budget.

The controller is deliberately pure python + JSON so a full dry-run of the
probe cycle can execute in CI with a fake workspace and a fake evaluator.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from dexpbt_isaaclab.msa.decision import (
    RetentionResult,
    RoutingResult,
    SeedEvalResult,
    decide_promotion,
    l_mid,
    panel_bootstrap_lower_quantile,
    route_second_half,
    same_tau_retention,
)
from dexpbt_isaaclab.msa.provenance import (
    ExposureSegment,
    InterventionRecord,
    MSAProvenance,
    probe_due,
    tau_plus_of,
)
from dexpbt_isaaclab.msa.state_machine import MSAPhase, MSAProbeStateMachine

PROBE_PACKET_FRAMES = 8_000_000        # each half-packet, worker-local frames
FORMAL_TOP_K = 3


@dataclass
class WorkerView:
    """The controller's read-only view of one live worker."""
    worker_id: int
    lineage_id: Optional[str]
    active_tau: float
    frame: int                          # worker-local monotonic ledger
    source_checkpoint_id: str
    is_floor_worker: bool = False


# Evaluator callback signature: (worker_id, snapshot_id, tau) -> SeedEvalResult.
# In production this is eval_bridge-based (subprocess ccrs_eval over the
# ordered seed manifest); in tests/dry-run it is a fake.
EvalFn = Callable[[int, str, float], SeedEvalResult]
# Snapshot callback: (worker_id, tag) -> snapshot_id. Production freezes the
# latest step-boundary checkpoint read-only; tests return labels.
SnapshotFn = Callable[[int, str], str]


@dataclass
class ProbeCycleResult:
    intervention_id: str
    due_workers: list[int]
    routing: Optional[RoutingResult]
    l_mid_by_worker: dict[int, float]
    panel_stat: Optional[dict]
    retention_by_worker: dict[int, dict]
    promoted: bool
    first_blocking_gate: str
    records: dict[int, dict] = field(default_factory=dict)


class MSAController:
    """Drives one probe cycle for the whole panel of due workers.

    state_dir holds the per-worker control-state JSONs (the existing
    curriculum_control/ directory in a live run; a temp dir in tests).
    """

    def __init__(
        self,
        state_dir: Path,
        evaluate: EvalFn,
        snapshot: SnapshotFn,
        *,
        retention_threshold: float,
        evaluator_manifest_id: str,
        top_k: int = FORMAL_TOP_K,
        packet_frames: int = PROBE_PACKET_FRAMES,
        permuted_route: bool = False,
        set_forced_tolerance: Optional[Callable[[int, float], None]] = None,
        train_frames: Optional[Callable[[int, int, float], int]] = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.evaluate = evaluate
        self.snapshot = snapshot
        self.retention_threshold = float(retention_threshold)
        self.evaluator_manifest_id = evaluator_manifest_id
        self.top_k = int(top_k)
        self.packet_frames = int(packet_frames)
        self.permuted_route = bool(permuted_route)
        # dependency-injected side effects (real launcher wires these to the
        # control JSON + supervisor; dry-run wires them to a simulator)
        self._set_forced_tolerance = set_forced_tolerance or (lambda w, t: None)
        self._train_frames = train_frames or (lambda w, n, t: n)
        self.machine = MSAProbeStateMachine()

    # -- provenance persistence (rides in the control-state chain) ----------
    def _prov_path(self, worker_id: int) -> Path:
        return self.state_dir / f"msa_provenance_{worker_id:03d}.json"

    def load_provenance(self, worker_id: int) -> Optional[MSAProvenance]:
        p = self._prov_path(worker_id)
        if not p.is_file():
            return None
        try:
            return MSAProvenance.from_dict(json.loads(p.read_text()))
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def save_provenance(self, prov: MSAProvenance) -> None:
        p = self._prov_path(prov.worker_id)
        tmp = p.parent / f"{p.stem}.{uuid.uuid4().hex[:8]}.tmp"
        tmp.write_text(json.dumps(prov.to_dict(), indent=2))
        tmp.replace(p)

    # -- due computation ------------------------------------------------------
    def due_workers(self, workers: list[WorkerView]) -> list[int]:
        due = []
        for w in workers:
            prov = self.load_provenance(w.worker_id)
            if probe_due(prov, w.frame, w.active_tau,
                         is_floor_worker=w.is_floor_worker):
                due.append(w.worker_id)
        return sorted(due)

    # -- one full probe cycle -------------------------------------------------
    def run_cycle(self, workers: list[WorkerView]) -> ProbeCycleResult:
        by_id = {w.worker_id: w for w in workers}
        due = self.due_workers(workers)
        intervention_id = f"msa_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        if not due:
            return ProbeCycleResult(
                intervention_id=intervention_id, due_workers=[],
                routing=None, l_mid_by_worker={}, panel_stat=None,
                retention_by_worker={}, promoted=False,
                first_blocking_gate="no_due_workers")

        # All due workers must share the same active tau for a common probe
        # (the deadlock band is a shared plateau); mixed-tau panels are run
        # per-tau by the caller.
        taus = {round(by_id[w].active_tau, 9) for w in due}
        if len(taus) != 1:
            raise ValueError(f"due workers span multiple taus {taus}; "
                             "run one cycle per tau")
        tau = by_id[due[0]].active_tau
        tau_plus = tau_plus_of(tau)

        records: dict[int, InterventionRecord] = {}
        provs: dict[int, MSAProvenance] = {}
        for w in due:
            wv = by_id[w]
            prov = self.load_provenance(w) or MSAProvenance(worker_id=w)
            provs[w] = prov
            records[w] = InterventionRecord(
                intervention_id=intervention_id, worker_id=w,
                lineage_id=wv.lineage_id,
                source_checkpoint_id=wv.source_checkpoint_id,
                active_tau=tau, proposed_tau=tau_plus, due_frame=wv.frame,
                evaluator_manifest_id=self.evaluator_manifest_id)

        m = self.machine

        # -- PRE_EVAL: snapshot + p_pre(tau), q_pre(tau_plus) ------------------
        m.transition(MSAPhase.PRE_EVAL, intervention_id)
        m.intervention_id = intervention_id
        pre_tau: dict[int, SeedEvalResult] = {}
        pre_plus: dict[int, SeedEvalResult] = {}
        for w in due:
            snap = self.snapshot(w, "pre")
            records[w].snapshot_pre_id = snap
            pre_tau[w] = self.evaluate(w, snap, tau)
            pre_plus[w] = self.evaluate(w, snap, tau_plus)

        # -- COMMON_PROBE: everyone trains packet_frames at tau_plus ----------
        m.transition(MSAPhase.COMMON_PROBE)
        for w in due:
            self._set_forced_tolerance(w, tau_plus)
            start = by_id[w].frame
            end = self._train_frames(w, self.packet_frames, tau_plus)
            if end - start != self.packet_frames:
                raise RuntimeError(
                    f"worker {w}: common probe trained {end - start} frames, "
                    f"expected exactly {self.packet_frames}")
            by_id[w].frame = end
            seg = ExposureSegment("common_probe", tau_plus, start, end,
                                  intervention_id)
            records[w].add_segment(seg)
            provs[w].record_exposure(seg)

        # -- MID_EVAL: q_mid(tau_plus), L_mid ---------------------------------
        m.transition(MSAPhase.MID_EVAL)
        l_mid_by_worker: dict[int, float] = {}
        mid_plus: dict[int, SeedEvalResult] = {}
        for w in due:
            snap = self.snapshot(w, "mid")
            records[w].snapshot_mid_id = snap
            mid_plus[w] = self.evaluate(w, snap, tau_plus)
            l_mid_by_worker[w] = l_mid(pre_plus[w], mid_plus[w])

        # -- ROUTE_SECOND_HALF -------------------------------------------------
        m.transition(MSAPhase.ROUTE_SECOND_HALF)
        routing = route_second_half(l_mid_by_worker, top_k=self.top_k,
                                    permuted=self.permuted_route)
        for w in due:
            records[w].recipient_status = (
                ("permuted_recipient" if self.permuted_route else
                 "top3_recipient") if w in routing.recipients
                else "common_only")

        # -- SECOND_PROBE: recipients only -------------------------------------
        m.transition(MSAPhase.SECOND_PROBE)
        for w in routing.recipients:
            start = by_id[w].frame
            end = self._train_frames(w, self.packet_frames, tau_plus)
            if end - start != self.packet_frames:
                raise RuntimeError(
                    f"worker {w}: second probe trained {end - start} frames, "
                    f"expected exactly {self.packet_frames}")
            by_id[w].frame = end
            seg = ExposureSegment("second_probe", tau_plus, start, end,
                                  intervention_id)
            records[w].add_segment(seg)
            provs[w].record_exposure(seg)

        # -- POST_EVAL: tolerance back to tau; p_post(tau); q_post follow-up ---
        m.transition(MSAPhase.POST_EVAL)
        retention_by_worker: dict[int, RetentionResult] = {}
        for w in due:
            self._set_forced_tolerance(w, tau)   # revert; NO checkpoint reload
            snap = self.snapshot(w, "post")
            records[w].snapshot_post_id = snap
            post_tau = self.evaluate(w, snap, tau)
            retention_by_worker[w] = same_tau_retention(pre_tau[w], post_tau)
            if w in routing.recipients:
                # q_post: unequal-dose follow-up for recipients ONLY; stored
                # in the record, never merged into the common-dose statistic.
                q_post = self.evaluate(w, snap, tau_plus)
                records[w].__dict__.setdefault("q_post_mean", q_post.mean)

        # -- DECIDE -------------------------------------------------------------
        m.transition(MSAPhase.DECIDE)
        paired = {
            w: [mv - pv for pv, mv in
                zip(pre_plus[w].success, mid_plus[w].success)]
            for w in due
        }
        panel_stat = panel_bootstrap_lower_quantile(paired)
        decision = decide_promotion(panel_stat, retention_by_worker,
                                    self.retention_threshold)
        for w in due:
            records[w].first_blocking_gate = decision.first_blocking_gate
            records[w].final_decision = ("promote" if decision.promote
                                         else "stay")
            provs[w].record_intervention(records[w])
            self.save_provenance(provs[w])

        if decision.promote:
            m.transition(MSAPhase.PROMOTE, "all gates passed")
            for w in due:
                self._set_forced_tolerance(w, tau_plus)  # permanent promotion
            m.transition(MSAPhase.NORMAL, "promotion applied")
        else:
            m.transition(MSAPhase.NORMAL, decision.first_blocking_gate)

        return ProbeCycleResult(
            intervention_id=intervention_id, due_workers=due,
            routing=routing, l_mid_by_worker=l_mid_by_worker,
            panel_stat=panel_stat,
            retention_by_worker={w: vars(r)
                                 for w, r in retention_by_worker.items()},
            promoted=decision.promote,
            first_blocking_gate=decision.first_blocking_gate,
            records={w: vars(records[w]) for w in due})
