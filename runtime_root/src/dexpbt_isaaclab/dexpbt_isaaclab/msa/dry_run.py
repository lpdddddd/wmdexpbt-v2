"""Production-style in-memory dry-run of the MSA probe cycle (no Isaac/GPU).

Simulates six workers with HETEROGENEOUS live tolerances (taken from the
real 5B tips), a committed-tick-quantized parallel trainer (20M frames per
tick, exactly like the PBT ledger), phase-barrier holds (frames cannot
advance while a worker is held), a persistent control-state store (the
in-memory equivalent of CurriculumControlState.msa_*), pin/commit tolerance
semantics and optional crash injection per phase — so unit tests can drive
MSAController through NORMAL -> ... -> DECIDE -> NORMAL/PROMOTE, kill it in
any phase, restart it and verify idempotent resume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from dexpbt_isaaclab.msa.controller import (
    MSAController,
    PROBE_PACKET_FRAMES,
    ProbeCycleResult,
    WorkerView,
)
from dexpbt_isaaclab.msa.decision import SeedEvalResult
from dexpbt_isaaclab.msa.eval_bridge import (
    PILOT_NUM_SEEDS,
    build_seed_manifest,
    canonical_tau,
)
from dexpbt_isaaclab.msa.provenance import tau_plus_of

# Heterogeneous live tolerances of the six 5B tips (meters, full precision —
# deliberately NOT 6-decimal-friendly).
DRY_RUN_TAUS = {
    0: 0.023112487509668322,
    1: 0.022475470952392342,
    2: 0.026831565886643130,
    3: 0.024903568922318386,
    4: 0.026214283076124620,
    5: 0.027593982185394335,
}
TICK_FRAMES = 20_000_000            # committed PBT tick quantum
DRY_RUN_NUM_SEEDS = PILOT_NUM_SEEDS  # pilot N=16 (formal N=256, explicit)
DRY_RUN_RHO_THRESHOLD = 0.88


class InjectedCrash(RuntimeError):
    pass


class HardKill(BaseException):
    """Simulates kill -9 / power loss: bypasses the controller's cleanup
    (except Exception) so restart/resume behavior can be tested."""


@dataclass
class FakeWorld:
    """Deterministic fake trainer/evaluator/state store.

    - frames advance ONLY through train_until (tick-quantized, parallel);
    - held workers cannot advance (eval phase barrier);
    - pin/commit bookkeeping mirrors msa_forced_success_tolerance and the
      durable live tolerance + resume snapshot;
    - the state store survives across controller instances (restart tests).
    """
    manifest_id: str
    seeds: list[int]
    taus: dict[int, float] = field(default_factory=lambda: dict(DRY_RUN_TAUS))
    skill_tau: dict[int, float] = field(default_factory=dict)
    skill_plus: dict[int, float] = field(default_factory=dict)
    learn_rate: dict[int, float] = field(default_factory=dict)
    frame: dict[int, int] = field(default_factory=dict)
    live_tolerance: dict[int, float] = field(default_factory=dict)
    pinned_tolerance: dict[int, float] = field(default_factory=dict)
    resume_snapshot_tolerance: dict[int, float] = field(default_factory=dict)
    held: set[int] = field(default_factory=set)
    # persistent (prov, machine, cycle) store == control JSON msa_* fields
    state_store: dict[int, dict] = field(default_factory=dict)
    # bookkeeping for assertions
    trained_frames: dict[int, int] = field(default_factory=dict)
    trained_frames_at_tol: dict[tuple[int, str], int] = field(default_factory=dict)
    pin_log: list[tuple[int, float]] = field(default_factory=list)
    commit_log: list[tuple[int, float]] = field(default_factory=list)
    hold_log: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)
    eval_calls: list[tuple[int, str, str]] = field(default_factory=list)
    snapshots: dict[str, tuple[int, float, float]] = field(default_factory=dict)
    reset_calls: list[str] = field(default_factory=list)   # must stay empty
    frames_during_eval_guard: bool = True
    # crash injection: raise InjectedCrash when this phase-tagged action
    # runs; crash_hard=True raises HardKill (BaseException) instead so the
    # controller cleanup is bypassed (kill -9 simulation).
    crash_on: Optional[str] = None     # "pre"/"mid"/"post"/"eval"/"common_train"/"commit"
    crash_hard: bool = False

    def _boom(self, msg: str) -> None:
        if self.crash_hard:
            raise HardKill(msg)
        raise InjectedCrash(msg)

    # -- snapshots ------------------------------------------------------------
    def snapshot(self, worker_id: int, tag: str) -> str:
        if worker_id not in self.held:
            raise AssertionError(
                f"snapshot of worker {worker_id} while NOT held "
                "(policy could advance during eval)")
        snap_id = f"w{worker_id}_{tag}"
        if snap_id not in self.snapshots:      # idempotent reuse on resume
            self.snapshots[snap_id] = (worker_id, self.skill_tau[worker_id],
                                       self.skill_plus[worker_id])
        return snap_id

    # -- evaluation -----------------------------------------------------------
    def evaluate(self, worker_id: int, snapshot_id: str,
                 tau: float) -> SeedEvalResult:
        if self.crash_on == "eval" or (
                self.crash_on and snapshot_id.endswith(self.crash_on)):
            self._boom(f"injected eval crash at {snapshot_id}")
        if worker_id not in self.held:
            raise AssertionError(
                f"evaluation of worker {worker_id} while NOT held")
        _, s_tau, s_plus = self.snapshots[snapshot_id]
        w_tau = float(canonical_tau(self.taus[worker_id]))
        w_plus = float(canonical_tau(tau_plus_of(self.taus[worker_id])))
        if abs(tau - w_plus) < 1e-15:
            base, kind = s_plus, "tau_plus"
        elif abs(tau - w_tau) < 1e-15:
            base, kind = s_tau, "tau"
        else:
            raise AssertionError(
                f"worker {worker_id} evaluated at foreign tolerance {tau} "
                f"(local tau={w_tau}, tau_plus={w_plus})")
        self.eval_calls.append((worker_id, snapshot_id, kind))
        n = len(self.seeds)
        success = [min(1.0, max(0.0, base + ((i % 5) - 2) * 0.002))
                   for i in range(n)]
        return SeedEvalResult(manifest_id=self.manifest_id, tau=tau,
                              seeds=list(self.seeds), success=success)

    # -- tolerance pin / durable commit ---------------------------------------
    def pin_tolerance(self, worker_id: int, tol: float) -> None:
        self.pin_log.append((worker_id, tol))
        self.pinned_tolerance[worker_id] = float(tol)
        self.live_tolerance[worker_id] = float(tol)

    def commit_tolerance(self, worker_id: int, tol: float) -> None:
        if self.crash_on == "commit":
            self.crash_on = None    # crash once; cleanup commits then succeed
            self._boom("injected commit crash")
        if worker_id in self.held:
            # production commit waits for a live committed tick; a held
            # worker would deadlock — the controller must commit only
            # while the worker is running.
            raise AssertionError(
                f"durable commit for worker {worker_id} while held")
        self.commit_log.append((worker_id, tol))
        self.live_tolerance[worker_id] = float(tol)
        self.resume_snapshot_tolerance[worker_id] = float(tol)
        self.pinned_tolerance[worker_id] = 0.0     # pin cleared

    # -- holds (phase barrier) --------------------------------------------------
    def hold_workers(self, worker_ids: list[int]) -> None:
        self.hold_log.append(("hold", tuple(worker_ids)))
        self.held.update(worker_ids)

    def release_workers(self, worker_ids: list[int]) -> None:
        self.hold_log.append(("release", tuple(worker_ids)))
        self.held.difference_update(worker_ids)

    # -- parallel tick-quantized trainer ---------------------------------------
    def train_until(self, targets: dict[int, int]) -> dict[int, int]:
        if self.crash_on == "common_train" and targets:
            self._boom("injected crash during common probe")
        for w in targets:
            if w in self.held:
                raise AssertionError(f"train_until on held worker {w}")
        observed: dict[int, int] = {}
        pending = dict(targets)
        # PARALLEL: one tick advances EVERY unheld worker simultaneously.
        while pending:
            for w in list(self.frame.keys()):
                if w in self.held:
                    continue
                self._advance_one_tick(w)
            for w in list(pending.keys()):
                if self.frame[w] >= pending[w]:
                    observed[w] = self.frame[w]     # OBSERVED committed frame
                    del pending[w]
        return observed

    def _advance_one_tick(self, w: int) -> None:
        self.frame[w] += TICK_FRAMES
        self.trained_frames[w] = self.trained_frames.get(w, 0) + TICK_FRAMES
        tol = self.live_tolerance[w]
        w_plus = float(canonical_tau(tau_plus_of(self.taus[w])))
        kind = "tau_plus" if abs(tol - w_plus) < 1e-15 else "tau"
        key = (w, kind)
        self.trained_frames_at_tol[key] = (
            self.trained_frames_at_tol.get(key, 0) + TICK_FRAMES)
        if kind == "tau_plus":
            self.skill_plus[w] = min(
                1.0, self.skill_plus[w]
                + self.learn_rate[w] * (TICK_FRAMES / 20_000_000))

    # -- persistent state store (== CurriculumControlState.msa_*) --------------
    def save_state(self, worker_id: int, prov, machine, cycle) -> None:
        slot = self.state_store.setdefault(worker_id, {})
        if prov is not None:
            slot["msa_provenance"] = json.loads(json.dumps(prov))
        if machine is not None:
            slot["msa_state_machine"] = json.loads(json.dumps(machine))
        if cycle is not None:
            slot["msa_cycle"] = json.loads(json.dumps(cycle, default=str))

    def load_state(self, worker_id: int):
        slot = self.state_store.get(worker_id, {})
        return (slot.get("msa_provenance"), slot.get("msa_state_machine"),
                slot.get("msa_cycle"))


def make_world(*, permuted: bool = False,
               learn_rates: dict[int, float] | None = None,
               skill_tau: float = 0.55, skill_plus: float = 0.30,
               num_seeds: int = DRY_RUN_NUM_SEEDS,
               rho_threshold: float = DRY_RUN_RHO_THRESHOLD,
               taus: dict[int, float] | None = None,
               world: FakeWorld | None = None,
               crash_on: Optional[str] = None,
               ) -> tuple[FakeWorld, MSAController, list[WorkerView]]:
    manifest_id, seeds = build_seed_manifest(num_seeds)
    if world is None:
        world = FakeWorld(manifest_id=manifest_id, seeds=seeds)
        if taus:
            world.taus = dict(taus)
        lr = learn_rates or {0: 0.06, 1: 0.10, 2: 0.02, 3: 0.08,
                             4: 0.04, 5: 0.06}
        for w in sorted(world.taus.keys()):
            world.skill_tau[w] = skill_tau
            world.skill_plus[w] = skill_plus
            world.learn_rate[w] = lr[w]
            world.frame[w] = 5_000_000_000
            world.live_tolerance[w] = world.taus[w]
            world.pinned_tolerance[w] = 0.0
    world.crash_on = crash_on
    views = [
        WorkerView(worker_id=w, lineage_id=f"lin_{w}",
                   active_tau=world.taus[w], frame=world.frame[w],
                   source_checkpoint_id=f"ws/{w:03d}/000160.pth")
        for w in sorted(world.taus.keys())
    ]
    ctl = MSAController(
        evaluate=world.evaluate, snapshot=world.snapshot,
        retention_threshold=rho_threshold,
        evaluator_manifest_id=manifest_id, permuted_route=permuted,
        pin_tolerance=world.pin_tolerance,
        commit_tolerance=world.commit_tolerance,
        hold_workers=world.hold_workers,
        release_workers=world.release_workers,
        train_until=world.train_until,
        save_state=world.save_state,
        load_state=world.load_state,
        packet_frames=PROBE_PACKET_FRAMES)
    return world, ctl, views


def run_dry_run(*, permuted: bool = False,
                ) -> tuple[FakeWorld, MSAController, ProbeCycleResult]:
    world, ctl, views = make_world(permuted=permuted)
    result = ctl.run_cycle(views)
    return world, ctl, result


def format_trace(world: FakeWorld, ctl: MSAController,
                 result: ProbeCycleResult) -> str:
    lines = [
        "# MSA Probe Dry-Run Trace (production-style)",
        "",
        f"- scenario: 6 workers with HETEROGENEOUS live tolerances "
        f"(real 5B-tip values), tick-quantized parallel trainer "
        f"({TICK_FRAMES:,} frames per committed tick), phase-barrier "
        f"holds, N={len(world.seeds)} pilot seeds (formal N=256 explicit "
        f"only)",
        f"- intervention_id: `{result.intervention_id}`",
        f"- arm: {'MSA_PERMUTED_ROUTE' if ctl.permuted_route else 'MSA'}",
        "",
        "## Worker-local tolerances",
        "",
        "| worker | tau (m) | tau_plus (m) |",
        "|---|---|---|",
    ]
    for w in sorted(result.tau_by_worker):
        lines.append(f"| p{w} | {canonical_tau(result.tau_by_worker[w])} "
                     f"| {canonical_tau(result.tau_plus_by_worker[w])} |")
    lines += [
        "",
        "## State machine transitions",
        "",
    ]
    lines += [f"1. `{h}`" for h in ctl.machine.history]
    lines += [
        "",
        "## Common probe (parallel, one 20M committed tick each)",
        "",
        "| worker | L_mid (criterion-local) | tau_plus frames |",
        "|---|---|---|",
    ]
    for w in sorted(result.l_mid_by_worker):
        plus = world.trained_frames_at_tol.get((w, "tau_plus"), 0)
        lines.append(f"| p{w} | {result.l_mid_by_worker[w]:+.4f} "
                     f"| {plus:,} |")
    r = result.routing
    lines += [
        "",
        "## Second-half routing",
        "",
        f"- order (L_mid desc, tie=worker_id asc): {r.order}",
        f"- recipients (top-{len(r.recipients)}): {r.recipients}",
        f"- permutation_seed: {r.permutation_seed}",
        "",
        "## Retention (worker-local tau) and decision",
        "",
        f"- panel L_mid 5th-pct (criterion-local conditional panel): "
        f"{result.panel_stat['lower_quantile']:+.4f}",
        f"- promoted: {result.promoted}",
        f"- first_blocking_gate: {result.first_blocking_gate}",
        "",
        "| worker | tau | p_pre | p_post | rho | unavailable |",
        "|---|---|---|---|---|---|",
    ]
    for w in sorted(result.retention_by_worker):
        d = result.retention_by_worker[w]
        rho = "n/a" if d["rho"] is None else f"{d['rho']:.3f}"
        lines.append(f"| p{w} | {canonical_tau(d['tau'])} | "
                     f"{d['p_pre']:.3f} | {d['p_post']:.3f} "
                     f"| {rho} | {d['retention_unavailable']} |")
    total_plus = sum(v for (w, k), v in world.trained_frames_at_tol.items()
                     if k == "tau_plus")
    lines += [
        "",
        "## Invariants observed",
        "",
        f"- reset calls (policy/optimizer/replay/lineage/curriculum): "
        f"{len(world.reset_calls)} (must be 0)",
        f"- durable tolerance commits (pin cleared): {world.commit_log}",
        f"- residual pins after cycle: "
        f"{ {w: p for w, p in world.pinned_tolerance.items() if p} } "
        "(must be empty)",
        f"- total tau_plus dose: {total_plus:,} frames "
        f"(= 6 x {TICK_FRAMES:,} common + {len(r.recipients)} x "
        f"{TICK_FRAMES:,} second half)",
        "- frames never advanced while a worker was held (asserted in "
        "FakeWorld.snapshot/evaluate/train_until).",
        "- q_post evaluated for recipients only; never entered L_mid.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--permuted", action="store_true")
    args = ap.parse_args(argv)
    from pathlib import Path
    world, ctl, result = run_dry_run(permuted=args.permuted)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(format_trace(world, ctl, result))
    print(f"wrote {out}")
    print(json.dumps({"promoted": result.promoted,
                      "first_blocking_gate": result.first_blocking_gate,
                      "recipients": result.routing.recipients,
                      "tau_by_worker": result.tau_by_worker}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
