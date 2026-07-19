"""MSA probe controller: drives the state machine over the existing
curriculum control-state chain.

Integration contract (no trainer surgery):
- The env obeys the control JSON: `msa_forced_success_tolerance` (sidecar-
  owned, written ONLY through the locked update_curriculum_control API) PINS
  the live tolerance while a probe segment is active; the msa_* state fields
  ride in the same JSON (and therefore in the same resume snapshot chain).
- The controller runs as a SIDECAR process next to the training supervisors.
  It NEVER reloads checkpoints, never touches policy/optimizer/replay/PBT
  lineage, and never resets curriculum state: its only mutation channels are
  the tolerance pin/commit and the msa_* bookkeeping fields.
- Heterogeneous tolerances: every worker w keeps its own tau[w] (= live
  tolerance at cycle start) and tau_plus[w] = max(0.010, 0.95 * tau[w]).
  All pre/mid/post evaluations, pins, provenance records and retention
  rho[w] use the worker-local values. L_mid[w] = q_mid - q_pre at the
  worker's OWN tau_plus is therefore a CRITERION-LOCAL improvement; the
  cross-worker routing / panel statistic conditions on this panel of
  criterion-local improvements (documented in decision.py).
- Exposure is PARALLEL: all due workers are pinned to their tau_plus first,
  then a single train_until barrier waits for every worker's committed tick
  ledger to pass start + packet. Frames recorded in provenance are the
  OBSERVED committed frames (never a synthetic start + n).
- Eval phases are BARRIERS: participating workers are held (paused at a
  committed checkpoint) while their snapshots are frozen and evaluated, so
  training can never advance a policy that is being measured.
- Every phase transition persists (machine, cycle, provenance) into the
  per-worker CurriculumControlState via the locked writer; a sidecar
  restart resumes from the persisted context and never re-trains a
  completed segment (segments are keyed by intervention_id + type in the
  provenance; snapshots and evals are idempotent by construction).
- Any exception in any phase triggers cleanup: non-promoted workers are
  durably restored to their own tau, holds and pins are cleared, the cycle
  is marked FAILED (with the failing phase) and the machine is reset to
  NORMAL; the raw eval results remain for an idempotent re-run.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
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
from dexpbt_isaaclab.msa.eval_bridge import canonical_tau
from dexpbt_isaaclab.msa.provenance import (
    ExposureSegment,
    InterventionRecord,
    MSAProvenance,
    probe_due,
    tau_plus_of,
)
from dexpbt_isaaclab.msa.state_machine import MSAPhase, MSAProbeStateMachine

# One committed PBT tick is the smallest reliably observable training
# quantum (20M env frames at --pbt-interval-steps 20000000). The probe
# packet is therefore exactly one committed tick; provenance stores the
# frames actually observed on the ledger, which may exceed the packet by
# tick quantization but never fabricates start + n.
PROBE_PACKET_FRAMES = 20_000_000
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


# --- injected side-effect callbacks (production wiring in sidecar.py) -------
# evaluate(worker_id, snapshot_id, tau) -> SeedEvalResult   (idempotent)
EvalFn = Callable[[int, str, float], SeedEvalResult]
# snapshot(worker_id, tag) -> snapshot_id; tag carries the intervention id so
# the production wiring can reuse an already-frozen snapshot on resume.
SnapshotFn = Callable[[int, str], str]
# pin_tolerance(worker_id, tol): set msa_forced_success_tolerance (locked).
PinFn = Callable[[int, float], None]
# commit_tolerance(worker_id, tol): durably make tol the live tolerance
# (resume-snapshot synced) and CLEAR the pin. Used for stay-revert and for
# promotion. Idempotent.
CommitFn = Callable[[int, float], None]
# hold_workers(ids) / release_workers(ids): pause/resume training at a
# committed checkpoint boundary (idempotent, re-entrant).
HoldFn = Callable[[list[int]], None]
# train_until({worker: target_frame}) -> {worker: observed_frame}: PARALLEL
# barrier over the committed tick ledgers of all listed workers. Returns
# frames actually observed (>= target). Never trains a worker not listed.
TrainUntilFn = Callable[[dict[int, int]], dict[int, int]]
# persistence of (provenance?, machine?, cycle?) per worker into the control
# chain; load returns the persisted dicts (or None each).
SaveStateFn = Callable[[int, Optional[dict], Optional[dict], Optional[dict]], None]
LoadStateFn = Callable[[int], tuple[Optional[dict], Optional[dict], Optional[dict]]]


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
    tau_by_worker: dict[int, float] = field(default_factory=dict)
    tau_plus_by_worker: dict[int, float] = field(default_factory=dict)
    records: dict[int, dict] = field(default_factory=dict)
    resumed: bool = False


class CycleFailure(RuntimeError):
    """Raised after cleanup when a probe cycle failed; safe to re-run."""


class MSAController:
    """Drives one probe cycle for the whole panel of due workers."""

    def __init__(
        self,
        evaluate: EvalFn,
        snapshot: SnapshotFn,
        *,
        retention_threshold: float,
        evaluator_manifest_id: str,
        pin_tolerance: PinFn,
        commit_tolerance: CommitFn,
        hold_workers: HoldFn,
        release_workers: HoldFn,
        train_until: TrainUntilFn,
        save_state: SaveStateFn,
        load_state: LoadStateFn,
        top_k: int = FORMAL_TOP_K,
        packet_frames: int = PROBE_PACKET_FRAMES,
        permuted_route: bool = False,
    ) -> None:
        self.evaluate = evaluate
        self.snapshot = snapshot
        self.retention_threshold = float(retention_threshold)
        self.evaluator_manifest_id = evaluator_manifest_id
        self.top_k = int(top_k)
        self.packet_frames = int(packet_frames)
        self.permuted_route = bool(permuted_route)
        self._pin = pin_tolerance
        self._commit = commit_tolerance
        self._hold = hold_workers
        self._release = release_workers
        self._train_until = train_until
        self._save_state = save_state
        self._load_state = load_state
        self.machine = MSAProbeStateMachine()

    # -- persisted state helpers ---------------------------------------------
    def load_provenance(self, worker_id: int) -> Optional[MSAProvenance]:
        prov, _, _ = self._load_state(worker_id)
        return MSAProvenance.from_dict(prov)

    def _persist(self, worker_ids: list[int],
                 provs: dict[int, MSAProvenance],
                 cycle: dict) -> None:
        machine_dict = self.machine.to_dict()
        for w in worker_ids:
            self._save_state(w, provs[w].to_dict(), machine_dict, dict(cycle))

    def pending_cycle(self, worker_ids: list[int]) -> Optional[dict]:
        """The persisted in-flight cycle context (for resume), if any.

        Returns the cycle dict of the most advanced worker whose persisted
        machine phase is mid-cycle and whose cycle is not terminal.
        """
        best: Optional[tuple[int, dict]] = None
        for w in worker_ids:
            _, machine_d, cycle_d = self._load_state(w)
            if not machine_d or not cycle_d:
                continue
            phase = str(machine_d.get("phase", "NORMAL"))
            status = str(cycle_d.get("status", ""))
            if phase == "NORMAL" or status in ("DONE", "FAILED"):
                continue
            hist_len = len(machine_d.get("history", []))
            if best is None or hist_len > best[0]:
                best = (hist_len, cycle_d)
        return best[1] if best else None

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
    def run_cycle(self, workers: list[WorkerView],
                  resume_cycle: Optional[dict] = None) -> ProbeCycleResult:
        by_id = {w.worker_id: w for w in workers}

        if resume_cycle is not None:
            intervention_id = str(resume_cycle["intervention_id"])
            due = [int(w) for w in resume_cycle["due_workers"]]
            tau = {int(k): float(v)
                   for k, v in resume_cycle["tau_by_worker"].items()}
            tau_plus = {int(k): float(v)
                        for k, v in resume_cycle["tau_plus_by_worker"].items()}
            start_frames = {int(k): int(v)
                            for k, v in resume_cycle["start_frames"].items()}
            resumed = True
        else:
            due = self.due_workers(workers)
            intervention_id = f"msa_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            if not due:
                return ProbeCycleResult(
                    intervention_id=intervention_id, due_workers=[],
                    routing=None, l_mid_by_worker={}, panel_stat=None,
                    retention_by_worker={}, promoted=False,
                    first_blocking_gate="no_due_workers")
            # HETEROGENEOUS tolerances: worker-local tau / tau_plus.
            tau = {w: float(canonical_tau(by_id[w].active_tau)) for w in due}
            tau_plus = {w: float(canonical_tau(tau_plus_of(tau[w])))
                        for w in due}
            start_frames = {w: int(by_id[w].frame) for w in due}
            resumed = False

        cycle: dict = dict(resume_cycle or {})
        cycle.update({
            "intervention_id": intervention_id,
            "due_workers": list(due),
            "tau_by_worker": {str(w): tau[w] for w in due},
            "tau_plus_by_worker": {str(w): tau_plus[w] for w in due},
            "start_frames": {str(w): start_frames[w] for w in due},
            "status": "RUNNING",
            "packet_frames": self.packet_frames,
            "evaluator_manifest_id": self.evaluator_manifest_id,
        })

        provs: dict[int, MSAProvenance] = {}
        records: dict[int, InterventionRecord] = {}
        for w in due:
            prov = self.load_provenance(w) or MSAProvenance(worker_id=w)
            provs[w] = prov
            records[w] = InterventionRecord(
                intervention_id=intervention_id, worker_id=w,
                lineage_id=by_id[w].lineage_id if w in by_id else None,
                source_checkpoint_id=(by_id[w].source_checkpoint_id
                                      if w in by_id else ""),
                active_tau=tau[w], proposed_tau=tau_plus[w],
                due_frame=start_frames[w],
                evaluator_manifest_id=self.evaluator_manifest_id)

        def _has_segment(w: int, segment_type: str) -> bool:
            return any(
                e.get("intervention_id") == intervention_id
                and e.get("segment_type") == segment_type
                for e in provs[w].exposures
            )

        held: set[int] = set()

        def _hold(ids: list[int]) -> None:
            self._hold(ids)
            held.update(ids)

        def _release(ids: list[int]) -> None:
            self._release(ids)
            held.difference_update(ids)

        m = self.machine
        promoted_committed = False
        try:
            # -- PRE_EVAL: barrier-hold, snapshot, p_pre(tau[w]), q_pre --------
            m.transition(MSAPhase.PRE_EVAL, intervention_id)
            m.intervention_id = intervention_id
            self._persist(due, provs, cycle)
            _hold(due)
            pre_tau: dict[int, SeedEvalResult] = {}
            pre_plus: dict[int, SeedEvalResult] = {}
            for w in due:
                snap = self.snapshot(w, f"{intervention_id}_pre")
                records[w].snapshot_pre_id = snap
                cycle.setdefault("snapshots", {}).setdefault(str(w), {})["pre"] = snap
                pre_tau[w] = self.evaluate(w, snap, tau[w])
                pre_plus[w] = self.evaluate(w, snap, tau_plus[w])
            # pin every due worker to its OWN tau_plus BEFORE anyone resumes,
            # so the common exposure starts simultaneously for the panel.
            for w in due:
                self._pin(w, tau_plus[w])
            self._persist(due, provs, cycle)

            # -- COMMON_PROBE: parallel packet at worker-local tau_plus --------
            m.transition(MSAPhase.COMMON_PROBE)
            self._persist(due, provs, cycle)
            todo = [w for w in due if not _has_segment(w, "common_probe")]
            _release(todo)
            observed = self._train_until(
                {w: start_frames[w] + self.packet_frames for w in todo})
            _hold(due)
            for w in todo:
                end = int(observed[w])
                if end < start_frames[w] + self.packet_frames:
                    raise RuntimeError(
                        f"worker {w}: common probe barrier returned frame "
                        f"{end} < start + packet "
                        f"({start_frames[w] + self.packet_frames})")
                seg = ExposureSegment("common_probe", tau_plus[w],
                                      start_frames[w], end, intervention_id)
                records[w].add_segment(seg)
                provs[w].record_exposure(seg)
            mid_frames = {
                w: max((int(e["segment_end_frame"])
                        for e in provs[w].exposures
                        if e.get("intervention_id") == intervention_id
                        and e.get("segment_type") == "common_probe"),
                       default=start_frames[w])
                for w in due
            }
            cycle["mid_frames"] = {str(w): mid_frames[w] for w in due}
            self._persist(due, provs, cycle)

            # -- MID_EVAL: q_mid at worker-local tau_plus; L_mid --------------
            m.transition(MSAPhase.MID_EVAL)
            self._persist(due, provs, cycle)
            l_mid_by_worker: dict[int, float] = {}
            mid_plus: dict[int, SeedEvalResult] = {}
            for w in due:
                snap = self.snapshot(w, f"{intervention_id}_mid")
                records[w].snapshot_mid_id = snap
                cycle.setdefault("snapshots", {}).setdefault(str(w), {})["mid"] = snap
                mid_plus[w] = self.evaluate(w, snap, tau_plus[w])
                # criterion-local: pre/mid pair shares the worker's OWN
                # tau_plus and manifest; l_mid() fails closed otherwise.
                l_mid_by_worker[w] = l_mid(pre_plus[w], mid_plus[w])

            # -- ROUTE_SECOND_HALF ---------------------------------------------
            m.transition(MSAPhase.ROUTE_SECOND_HALF)
            routing = route_second_half(l_mid_by_worker, top_k=self.top_k,
                                        permuted=self.permuted_route)
            for w in due:
                records[w].recipient_status = (
                    ("permuted_recipient" if self.permuted_route else
                     "top3_recipient") if w in routing.recipients
                    else "common_only")
            cycle["recipients"] = list(routing.recipients)
            cycle["l_mid_by_worker"] = {str(w): l_mid_by_worker[w]
                                        for w in due}
            # non-recipients are re-pinned to their OWN tau IMMEDIATELY (a
            # pin write is synchronous and safe while held; the DURABLE
            # commit happens after the final release, because committing
            # needs the worker running to confirm a live tick). Recipients
            # stay pinned at their tau_plus.
            for w in due:
                if w not in routing.recipients:
                    self._pin(w, tau[w])
            self._persist(due, provs, cycle)

            # -- SECOND_PROBE: recipients only, parallel -----------------------
            m.transition(MSAPhase.SECOND_PROBE)
            self._persist(due, provs, cycle)
            second_todo = [w for w in routing.recipients
                           if not _has_segment(w, "second_probe")]
            # everyone resumes training (non-recipients at their tau); the
            # barrier waits only on the recipients' packet.
            _release(due)
            observed2 = self._train_until(
                {w: mid_frames[w] + self.packet_frames for w in second_todo})
            _hold(due)
            for w in second_todo:
                end = int(observed2[w])
                if end < mid_frames[w] + self.packet_frames:
                    raise RuntimeError(
                        f"worker {w}: second probe barrier returned frame "
                        f"{end} < mid + packet "
                        f"({mid_frames[w] + self.packet_frames})")
                seg = ExposureSegment("second_probe", tau_plus[w],
                                      mid_frames[w], end, intervention_id)
                records[w].add_segment(seg)
                provs[w].record_exposure(seg)
            self._persist(due, provs, cycle)

            # -- POST_EVAL: pins back at tau[w] (held); p_post(tau[w]) ---------
            m.transition(MSAPhase.POST_EVAL)
            self._persist(due, provs, cycle)
            retention_by_worker: dict[int, RetentionResult] = {}
            for w in due:
                self._pin(w, tau[w])      # revert pin; NO checkpoint reload
                snap = self.snapshot(w, f"{intervention_id}_post")
                records[w].snapshot_post_id = snap
                cycle.setdefault("snapshots", {}).setdefault(str(w), {})["post"] = snap
                post_tau = self.evaluate(w, snap, tau[w])
                retention_by_worker[w] = same_tau_retention(pre_tau[w], post_tau)
                if w in routing.recipients:
                    # q_post: unequal-dose follow-up for recipients ONLY;
                    # stored in the record, never merged into L_mid.
                    q_post = self.evaluate(w, snap, tau_plus[w])
                    records[w].__dict__.setdefault("q_post_mean", q_post.mean)
            self._persist(due, provs, cycle)

            # -- DECIDE ---------------------------------------------------------
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

            final_tol = dict(tau)
            if decision.promote:
                m.transition(MSAPhase.PROMOTE, "all gates passed")
                final_tol = dict(tau_plus)
                for w in due:
                    self._pin(w, tau_plus[w])
                m.transition(MSAPhase.NORMAL, "promotion applied")
            else:
                m.transition(MSAPhase.NORMAL, decision.first_blocking_gate)

            cycle["status"] = "DONE"
            cycle["decision"] = ("promote" if decision.promote else "stay")
            cycle["first_blocking_gate"] = decision.first_blocking_gate
            self._persist(due, provs, cycle)
            # durable commits need the workers RUNNING (a committed tick must
            # confirm the live tolerance before the pin is cleared and the
            # resume snapshot is synced), so release first, then commit.
            _release(due)
            for w in due:
                self._commit(w, final_tol[w])
            promoted_committed = decision.promote

            return ProbeCycleResult(
                intervention_id=intervention_id, due_workers=due,
                routing=routing, l_mid_by_worker=l_mid_by_worker,
                panel_stat=panel_stat,
                retention_by_worker={w: vars(r)
                                     for w, r in retention_by_worker.items()},
                promoted=decision.promote,
                first_blocking_gate=decision.first_blocking_gate,
                tau_by_worker=dict(tau), tau_plus_by_worker=dict(tau_plus),
                records={w: vars(records[w]) for w in due},
                resumed=resumed)

        except Exception as err:
            # -- crash cleanup: nobody may be left running at tau_plus ---------
            failed_phase = m.phase.value
            # 1. synchronous pin-revert first (effective at the next tick
            #    even if the durable commit below cannot complete);
            for w in due:
                if not promoted_committed:
                    try:
                        self._pin(w, tau[w])
                    except Exception:
                        pass
            # 2. clear all holds (a durable commit needs running workers);
            try:
                _release(sorted(held))
            except Exception:
                pass
            # 3. durable revert + pin clear, best effort.
            for w in due:
                if not promoted_committed:
                    try:
                        self._commit(w, tau[w])
                    except Exception:
                        pass
            cycle["status"] = "FAILED"
            cycle["failed_phase"] = failed_phase
            cycle["error"] = f"{type(err).__name__}: {err}"
            m.abort(f"{failed_phase}: {type(err).__name__}")
            for w in due:
                records[w].final_decision = "abort"
                records[w].first_blocking_gate = f"cycle_failed_{failed_phase}"
                provs[w].record_intervention(records[w])
            try:
                self._persist(due, provs, cycle)
            except Exception:
                pass
            raise CycleFailure(
                f"MSA cycle {intervention_id} failed in {failed_phase}: "
                f"{err}") from err
