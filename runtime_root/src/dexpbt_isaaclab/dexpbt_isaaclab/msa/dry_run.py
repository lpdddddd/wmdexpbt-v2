"""In-memory dry-run of a full MSA probe cycle (no Isaac, no GPU).

Simulates six workers stuck at tau=0.0235 with a deterministic fake
evaluator and fake trainer, drives MSAController through
NORMAL -> ... -> DECIDE -> NORMAL/PROMOTE, and returns a full trace
(used by DRY_RUN_TRACE.md generation and by the unit tests).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from dexpbt_isaaclab.msa.controller import (
    MSAController,
    ProbeCycleResult,
    WorkerView,
)
from dexpbt_isaaclab.msa.decision import SeedEvalResult
from dexpbt_isaaclab.msa.eval_bridge import build_seed_manifest
from dexpbt_isaaclab.msa.provenance import tau_plus_of

DRY_RUN_TAU = 0.0235
DRY_RUN_NUM_SEEDS = 16          # small N for dry-run/tests (formal N=256)
DRY_RUN_RHO_THRESHOLD = 0.88


@dataclass
class FakeWorld:
    """Deterministic fake trainer + evaluator with per-worker learning rates."""
    manifest_id: str
    seeds: list[int]
    # per-worker skill at (tau, tau_plus); probe training improves tau_plus
    skill_tau: dict[int, float] = field(default_factory=dict)
    skill_plus: dict[int, float] = field(default_factory=dict)
    learn_rate: dict[int, float] = field(default_factory=dict)
    trained_frames: dict[int, int] = field(default_factory=dict)
    forced_tol_log: list[tuple[int, float]] = field(default_factory=list)
    snapshots: dict[str, tuple[int, float, float]] = field(default_factory=dict)
    reset_calls: list[str] = field(default_factory=list)   # must stay empty
    frame: dict[int, int] = field(default_factory=dict)

    def snapshot(self, worker_id: int, tag: str) -> str:
        snap_id = f"w{worker_id}_{tag}"
        self.snapshots[snap_id] = (worker_id, self.skill_tau[worker_id],
                                   self.skill_plus[worker_id])
        return snap_id

    def evaluate(self, worker_id: int, snapshot_id: str,
                 tau: float) -> SeedEvalResult:
        _, s_tau, s_plus = self.snapshots[snapshot_id]
        base = s_plus if abs(tau - tau_plus_of(DRY_RUN_TAU)) < 1e-12 else s_tau
        # deterministic per-seed jitter, mean == base
        n = len(self.seeds)
        success = [min(1.0, max(0.0, base + ((i % 5) - 2) * 0.002))
                   for i in range(n)]
        return SeedEvalResult(manifest_id=self.manifest_id, tau=tau,
                              seeds=list(self.seeds), success=success)

    def set_forced_tolerance(self, worker_id: int, tol: float) -> None:
        self.forced_tol_log.append((worker_id, tol))

    def train_frames(self, worker_id: int, n_frames: int, tau: float) -> int:
        # training at tau_plus improves tau_plus skill; ledger is monotonic
        self.trained_frames[worker_id] = (
            self.trained_frames.get(worker_id, 0) + n_frames)
        self.skill_plus[worker_id] = min(
            1.0, self.skill_plus[worker_id]
            + self.learn_rate[worker_id] * (n_frames / 8_000_000))
        self.frame[worker_id] += n_frames
        return self.frame[worker_id]


def make_world(state_dir: Path, *, permuted: bool = False,
               learn_rates: dict[int, float] | None = None,
               skill_tau: float = 0.55, skill_plus: float = 0.30,
               num_seeds: int = DRY_RUN_NUM_SEEDS,
               rho_threshold: float = DRY_RUN_RHO_THRESHOLD,
               ) -> tuple[FakeWorld, MSAController, list[WorkerView]]:
    manifest_id, seeds = build_seed_manifest(num_seeds)
    world = FakeWorld(manifest_id=manifest_id, seeds=seeds)
    lr = learn_rates or {0: 0.06, 1: 0.10, 2: 0.02, 3: 0.08, 4: 0.04, 5: 0.06}
    views = []
    for w in range(6):
        world.skill_tau[w] = skill_tau
        world.skill_plus[w] = skill_plus
        world.learn_rate[w] = lr[w]
        world.frame[w] = 5_000_000_000
        views.append(WorkerView(
            worker_id=w, lineage_id=f"lin_{w}", active_tau=DRY_RUN_TAU,
            frame=world.frame[w],
            source_checkpoint_id=f"ws/00{w}/000160.pth"))
    ctl = MSAController(
        state_dir=state_dir, evaluate=world.evaluate,
        snapshot=world.snapshot, retention_threshold=rho_threshold,
        evaluator_manifest_id=manifest_id, permuted_route=permuted,
        set_forced_tolerance=world.set_forced_tolerance,
        train_frames=world.train_frames)
    return world, ctl, views


def run_dry_run(state_dir: Path, *, permuted: bool = False,
                ) -> tuple[FakeWorld, MSAController, ProbeCycleResult]:
    world, ctl, views = make_world(state_dir, permuted=permuted)
    result = ctl.run_cycle(views)
    return world, ctl, result


def format_trace(world: FakeWorld, ctl: MSAController,
                 result: ProbeCycleResult) -> str:
    tau_plus = tau_plus_of(DRY_RUN_TAU)
    lines = [
        "# MSA Probe Dry-Run Trace",
        "",
        f"- scenario: 6 workers deadlocked at tau={DRY_RUN_TAU} m "
        f"(tau_plus={tau_plus:.6f} m), fake deterministic evaluator/trainer, "
        f"N={DRY_RUN_NUM_SEEDS} seeds (formal N=256)",
        f"- intervention_id: `{result.intervention_id}`",
        f"- arm: {'MSA_PERMUTED_ROUTE' if ctl.permuted_route else 'MSA'}",
        "",
        "## State machine transitions",
        "",
    ]
    lines += [f"1. `{h}`" for h in ctl.machine.history]
    lines += [
        "",
        "## Common probe (first half, all due workers, 8M frames each)",
        "",
        "| worker | L_mid | frames trained |",
        "|---|---|---|",
    ]
    for w in sorted(result.l_mid_by_worker):
        lines.append(f"| p{w} | {result.l_mid_by_worker[w]:+.4f} "
                     f"| {world.trained_frames.get(w, 0):,} |")
    r = result.routing
    lines += [
        "",
        "## Second-half routing",
        "",
        f"- order (L_mid desc, tie=worker_id asc): {r.order}",
        f"- recipients (top-{len(r.recipients)}): {r.recipients}",
        f"- permutation_seed: {r.permutation_seed}",
        "",
        "## Retention (same tau) and decision",
        "",
        f"- panel L_mid 5th-pct (bootstrap, fixed seed): "
        f"{result.panel_stat['lower_quantile']:+.4f}",
        f"- promoted: {result.promoted}",
        f"- first_blocking_gate: {result.first_blocking_gate}",
        "",
        "| worker | p_pre | p_post | rho | unavailable |",
        "|---|---|---|---|---|",
    ]
    for w in sorted(result.retention_by_worker):
        d = result.retention_by_worker[w]
        rho = "n/a" if d["rho"] is None else f"{d['rho']:.3f}"
        lines.append(f"| p{w} | {d['p_pre']:.3f} | {d['p_post']:.3f} "
                     f"| {rho} | {d['retention_unavailable']} |")
    lines += [
        "",
        "## Invariants observed",
        "",
        f"- reset calls (policy/optimizer/replay/lineage/curriculum): "
        f"{len(world.reset_calls)} (must be 0)",
        f"- forced tolerance writes: {world.forced_tol_log}",
        f"- total probe frames charged to ledgers: "
        f"{sum(world.trained_frames.values()):,} "
        f"(= 6x8M common + {len(r.recipients)}x8M second half)",
        "- q_post evaluated for recipients only; never entered L_mid.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import tempfile

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--permuted", action="store_true")
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as td:
        world, ctl, result = run_dry_run(Path(td), permuted=args.permuted)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(format_trace(world, ctl, result))
    print(f"wrote {out}")
    print(json.dumps({"promoted": result.promoted,
                      "first_blocking_gate": result.first_blocking_gate,
                      "recipients": result.routing.recipients}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
