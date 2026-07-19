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

import threading
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
# quantum. The base --pbt-interval-steps is 20M env frames, but every 5B
# continuation runs in consolidation stage 3 (SELECTION, frame >= 3.5B)
# whose interval multiplier is 4.0 -> the REAL committed cadence is 80M
# env frames per tick (measured in the strict-5B source run and confirmed
# by the 2026-07-19 execution smoke: first commit at baseline+80M). The
# probe packet is exactly one committed tick at that real cadence;
# provenance stores the frames actually observed on the ledger, which may
# exceed the packet by tick quantization but never fabricates start + n.
PROBE_PACKET_FRAMES = 80_000_000
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


class CommitTimeout(TimeoutError):
    """A durable tolerance commit did not confirm within its bounded wait.

    Carries enough context for the sidecar to write COMMIT_TIMEOUT.json and
    stop the arm. The controller's crash cleanup MUST NOT start a second
    full bounded wait after this (the synchronous pin-revert is enough)."""

    def __init__(self, worker_id: int, target_tau: float,
                 last_frame: int, last_tau: float, max_wait_s: float) -> None:
        super().__init__(
            f"worker {worker_id}: live tolerance did not reach "
            f"{target_tau} within {max_wait_s}s (last committed frame "
            f"{last_frame}, last live tau {last_tau})")
        self.worker_id = int(worker_id)
        self.target_tau = float(target_tau)
        self.last_frame = int(last_frame)
        self.last_tau = float(last_tau)
        self.max_wait_s = float(max_wait_s)


# --- injected side-effect callbacks (production wiring in sidecar.py) -------
# evaluate(worker_id, snapshot_id, tau) -> SeedEvalResult   (idempotent)
EvalFn = Callable[[int, str, float], SeedEvalResult]
# snapshot(worker_id, tag) -> (snapshot_id, committed_frame); tag carries the
# intervention id so the production wiring can reuse an already-frozen
# snapshot on resume (returning the frame RECORDED at freeze time, never a
# re-read). The controller checks the frame against the phase boundary.
SnapshotFn = Callable[[int, str], tuple[str, int]]
# refresh_view(worker_id) -> WorkerView from the CURRENT committed tick.
# Called after a hold so packet start frames / snapshot frames can never be
# one tick stale (the due-detection -> hold race).
RefreshFn = Callable[[int], "WorkerView"]
# set_busy(info|None): maintain the MSA_BUSY.json coordination marker (the
# continuation guard defers park/cap actions and the watchdog treats a dead
# sidecar as fatal while it exists). Called with a dict BEFORE the first
# hold, refreshed at every phase persist, and with None only after the
# cycle is fully DONE or the failure cleanup completed.
SetBusyFn = Callable[[Optional[dict]], None]
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
        refresh_view: RefreshFn,
        set_busy: Optional[SetBusyFn] = None,
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
        self._refresh_view = refresh_view
        self._set_busy = set_busy or (lambda info: None)
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
        # keep the coordination marker in step with the persisted phase; it
        # exists from BEFORE the first hold until DONE / cleanup complete.
        self._set_busy({
            "intervention_id": str(cycle.get("intervention_id", "")),
            "phase": self.machine.phase.value,
            "status": str(cycle.get("status", "")),
            "due_workers": list(cycle.get("due_workers", [])),
            "timestamp": time.time(),
        })

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

    # -- durable commits (parallel, per-worker acks) ---------------------------
    def _commit_all(self, todo: list[int], final_tol: dict[int, float],
                    cycle: dict,
                    provs: dict[int, MSAProvenance]) -> None:
        """Durably commit every listed worker IN PARALLEL; persist each
        worker's ack the moment its commit confirms. Raises the first
        failure (CommitTimeout preferred) with the ack table attached —
        never a second bounded wait for an already-failed worker."""
        if not todo:
            return
        acks = cycle["commit_acks_by_worker"]
        lock = threading.Lock()
        failures: list[BaseException] = []

        def _one(w: int) -> None:
            try:
                self._commit(w, final_tol[w])
            except BaseException as exc:            # noqa: BLE001
                with lock:
                    failures.append(exc)
                return
            with lock:
                acks[str(w)] = True
                # ack persisted IMMEDIATELY (to every due worker's control,
                # so any surviving copy carries the same ack table)
                self._persist([int(k) for k in acks], provs, cycle)

        threads = [threading.Thread(target=_one, args=(w,), daemon=True)
                   for w in todo]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if failures:
            err = next((f for f in failures if isinstance(f, TimeoutError)),
                       failures[0])
            try:
                err.commit_acks = dict(acks)         # for COMMIT_TIMEOUT.json
            except Exception:
                pass
            raise err

    def _resume_committing(self, resume_cycle: dict) -> ProbeCycleResult:
        """Finish an interrupted COMMITTING cycle: NO re-eval, NO re-train —
        only idempotent durable commits for workers not yet acked."""
        cycle = dict(resume_cycle)
        intervention_id = str(cycle["intervention_id"])
        due = [int(w) for w in cycle["due_workers"]]
        tau = {int(k): float(v)
               for k, v in cycle["tau_by_worker"].items()}
        tau_plus = {int(k): float(v)
                    for k, v in cycle["tau_plus_by_worker"].items()}
        final_tol = {int(k): float(v)
                     for k, v in cycle["final_tol_by_worker"].items()}
        acks = {str(w): bool(cycle.get("commit_acks_by_worker", {})
                             .get(str(w), False)) for w in due}
        cycle["commit_acks_by_worker"] = acks
        # restore the persisted machine (PROMOTE for promote, DECIDE for
        # stay) so the final transition to NORMAL stays legal and audited.
        _, machine_d, _ = self._load_state(due[0])
        if machine_d:
            self.machine = MSAProbeStateMachine.from_dict(machine_d)
        m = self.machine
        provs = {w: self.load_provenance(w) or MSAProvenance(worker_id=w)
                 for w in due}
        promoted = cycle.get("decision") == "promote"
        try:
            self._persist(due, provs, cycle)        # re-arms the busy marker
            # workers are RUNNING here (holds died with the old process /
            # were released by the watchdog); commit only the un-acked.
            todo = [w for w in due if not acks[str(w)]]
            self._commit_all(todo, final_tol, cycle, provs)
            cycle["status"] = "DONE"
            m.transition(MSAPhase.NORMAL, "committing resumed")
            self._persist(due, provs, cycle)
            self._set_busy(None)
            return ProbeCycleResult(
                intervention_id=intervention_id, due_workers=due,
                routing=None,
                l_mid_by_worker={int(k): float(v) for k, v in
                                 cycle.get("l_mid_by_worker", {}).items()},
                panel_stat=None,
                retention_by_worker={int(k): v for k, v in
                                     cycle.get("retention_by_worker",
                                               {}).items()},
                promoted=promoted,
                first_blocking_gate=str(cycle.get("first_blocking_gate",
                                                  "")),
                tau_by_worker=dict(tau), tau_plus_by_worker=dict(tau_plus),
                resumed=True)
        except Exception as err:
            # promotion is NOT claimed; un-acked workers revert to their
            # own tau via a synchronous pin (never a second bounded wait
            # after a commit timeout).
            for w in due:
                if not acks[str(w)]:
                    try:
                        self._pin(w, tau[w])
                    except Exception:
                        pass
            cycle["status"] = "FAILED"
            cycle["failed_phase"] = m.phase.value
            cycle["error"] = f"{type(err).__name__}: {err}"
            m.abort(f"COMMITTING: {type(err).__name__}")
            try:
                self._persist(due, provs, cycle)
            except Exception:
                pass
            self._set_busy(None)
            raise CycleFailure(
                f"MSA cycle {intervention_id} failed while COMMITTING: "
                f"{err}") from err

    # -- one full probe cycle -------------------------------------------------
    def run_cycle(self, workers: list[WorkerView],
                  resume_cycle: Optional[dict] = None) -> ProbeCycleResult:
        by_id = {w.worker_id: w for w in workers}

        if (resume_cycle is not None
                and str(resume_cycle.get("status")) == "COMMITTING"):
            return self._resume_committing(resume_cycle)

        if resume_cycle is not None:
            intervention_id = str(resume_cycle["intervention_id"])
            due = [int(w) for w in resume_cycle["due_workers"]]
            tau = {int(k): float(v)
                   for k, v in resume_cycle["tau_by_worker"].items()}
            tau_plus = {int(k): float(v)
                        for k, v in resume_cycle["tau_plus_by_worker"].items()}
            # persisted packet boundaries are kept verbatim; the post-hold
            # read below only VALIDATES current frame >= persisted frame.
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
            # start_frames are set AFTER the PRE_EVAL hold from the freshly
            # re-read committed metas (a worker may commit one more tick
            # between due detection and the hold; that tick was trained at
            # tau and must NOT be counted as tau_plus exposure).
            start_frames = {}
            resumed = False

        cycle: dict = dict(resume_cycle or {})
        cycle.update({
            "intervention_id": intervention_id,
            "due_workers": list(due),
            "tau_by_worker": {str(w): tau[w] for w in due},
            "tau_plus_by_worker": {str(w): tau_plus[w] for w in due},
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
                due_frame=start_frames.get(
                    w, by_id[w].frame if w in by_id else 0),
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

        def _refresh_held(ids: list[int]) -> dict[int, WorkerView]:
            """Re-read the CURRENT committed tick of each held worker."""
            return {w: self._refresh_view(w) for w in ids}

        def _snapshot_at(w: int, tag: str, expected_frame: int) -> str:
            """Snapshot and enforce it comes from the phase-boundary frame."""
            snap, snap_frame = self.snapshot(w, tag)
            if int(snap_frame) != int(expected_frame):
                raise RuntimeError(
                    f"worker {w}: snapshot {snap} is at committed frame "
                    f"{snap_frame}, expected the phase boundary frame "
                    f"{expected_frame} (stale/foreign committed prefix)")
            return snap

        m = self.machine
        promoted_committed = False
        try:
            # -- PRE_EVAL: barrier-hold FIRST, then re-read committed metas ----
            m.transition(MSAPhase.PRE_EVAL, intervention_id)
            m.intervention_id = intervention_id
            self._persist(due, provs, cycle)
            _hold(due)
            # the packet start is the POST-hold committed frame: a tick that
            # committed between due detection and the hold ran at tau and is
            # excluded from the tau_plus exposure accounting.
            fresh = _refresh_held(due)
            for w in due:
                fv = fresh[w]
                if int(fv.worker_id) != int(w):
                    raise RuntimeError(f"refresh returned worker "
                                       f"{fv.worker_id} for {w}")
                live = canonical_tau(fv.active_tau)
                if resumed:
                    # a resumed cycle may find the worker still pinned at
                    # its own tau_plus (kill mid-cycle); anything else is a
                    # foreign tolerance and fails closed.
                    if live not in (canonical_tau(tau[w]),
                                    canonical_tau(tau_plus[w])):
                        raise RuntimeError(
                            f"worker {w}: live tolerance {live} is neither "
                            f"cycle tau {canonical_tau(tau[w])} nor "
                            f"tau_plus {canonical_tau(tau_plus[w])} on "
                            "resume")
                elif live != canonical_tau(tau[w]):
                    raise RuntimeError(
                        f"worker {w}: live tolerance {live} != cycle tau "
                        f"{canonical_tau(tau[w])} after hold (curriculum "
                        "moved between due detection and hold)")
                want_lin = (by_id[w].lineage_id if w in by_id else None)
                if (want_lin and fv.lineage_id
                        and str(fv.lineage_id) != str(want_lin)):
                    raise RuntimeError(
                        f"worker {w}: lineage {fv.lineage_id} != "
                        f"{want_lin} after hold")
                if resumed:
                    if int(fv.frame) < start_frames[w]:
                        raise RuntimeError(
                            f"worker {w}: committed frame {fv.frame} < "
                            f"persisted start frame {start_frames[w]} "
                            "(ledger regressed)")
                else:
                    start_frames[w] = int(fv.frame)
            cycle["start_frames"] = {str(w): start_frames[w] for w in due}
            pre_tau: dict[int, SeedEvalResult] = {}
            pre_plus: dict[int, SeedEvalResult] = {}
            for w in due:
                snap = _snapshot_at(w, f"{intervention_id}_pre",
                                    start_frames[w])
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
            # the segment end / MID snapshot frame is the POST-hold committed
            # frame (>= the barrier observation; a tick may commit between
            # the barrier and the hold and still belongs to this packet,
            # because the pin was active the whole time).
            fresh_mid = _refresh_held(todo)
            for w in todo:
                end = int(fresh_mid[w].frame)
                if end < int(observed[w]):
                    raise RuntimeError(
                        f"worker {w}: post-hold frame {end} < barrier "
                        f"observation {observed[w]} (ledger regressed)")
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
                snap = _snapshot_at(w, f"{intervention_id}_mid",
                                    mid_frames[w])
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
            fresh_post = _refresh_held(due)
            for w in second_todo:
                end = int(fresh_post[w].frame)
                if end < int(observed2[w]):
                    raise RuntimeError(
                        f"worker {w}: post-hold frame {end} < barrier "
                        f"observation {observed2[w]} (ledger regressed)")
                if end < mid_frames[w] + self.packet_frames:
                    raise RuntimeError(
                        f"worker {w}: second probe barrier returned frame "
                        f"{end} < mid + packet "
                        f"({mid_frames[w] + self.packet_frames})")
                seg = ExposureSegment("second_probe", tau_plus[w],
                                      mid_frames[w], end, intervention_id)
                records[w].add_segment(seg)
                provs[w].record_exposure(seg)
            post_frames = {w: int(fresh_post[w].frame) for w in due}
            self._persist(due, provs, cycle)

            # -- POST_EVAL: pins back at tau[w] (held); p_post(tau[w]) ---------
            m.transition(MSAPhase.POST_EVAL)
            self._persist(due, provs, cycle)
            retention_by_worker: dict[int, RetentionResult] = {}
            for w in due:
                self._pin(w, tau[w])      # revert pin; NO checkpoint reload
                snap = _snapshot_at(w, f"{intervention_id}_post",
                                    post_frames[w])
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

            # -- COMMITTING: persisted BEFORE any durable commit ---------------
            # The machine stays in PROMOTE (promote) / DECIDE (stay) and the
            # cycle is NOT DONE until every worker's durable commit is
            # acknowledged; a crash between DECIDE and the last ack resumes
            # into _resume_committing (commits only, no re-eval/re-train).
            cycle["status"] = "COMMITTING"
            cycle["decision"] = ("promote" if decision.promote else "stay")
            cycle["first_blocking_gate"] = decision.first_blocking_gate
            cycle["final_tol_by_worker"] = {str(w): final_tol[w] for w in due}
            cycle["commit_acks_by_worker"] = {str(w): False for w in due}
            cycle["retention_by_worker"] = {
                str(w): vars(r) for w, r in retention_by_worker.items()}
            self._persist(due, provs, cycle)
            # durable commits need the workers RUNNING (a committed tick must
            # confirm the live tolerance before the pin is cleared and the
            # resume snapshot is synced), so release first, then commit ALL
            # workers IN PARALLEL (serial waits would let later workers train
            # extra ticks while earlier ones wait for their ack).
            _release(due)
            self._commit_all(due, final_tol, cycle, provs)
            cycle["status"] = "DONE"
            m.transition(MSAPhase.NORMAL,
                         "promotion applied" if decision.promote
                         else decision.first_blocking_gate)
            self._persist(due, provs, cycle)
            self._set_busy(None)
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
            # 3. durable revert + pin clear, best effort — EXCEPT after a
            #    commit timeout: a durable commit is a bounded wait on live
            #    ticks, and cleanup must never start a second full timeout
            #    (the synchronous pin-revert above already restores tau at
            #    the next tick; the sidecar stops the arm).
            if not isinstance(err, TimeoutError):
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
            # cleanup is complete: only now may the busy marker go away
            # (a hard kill skips this whole handler and leaves it in place
            # for the watchdog / guard to act on).
            self._set_busy(None)
            raise CycleFailure(
                f"MSA cycle {intervention_id} failed in {failed_phase}: "
                f"{err}") from err
