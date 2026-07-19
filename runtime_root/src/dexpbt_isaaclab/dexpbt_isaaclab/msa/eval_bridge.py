"""Bridge from MSA probe phases to the existing CCRS policy-only evaluator.

The evaluator itself is dexpbt_isaaclab.scripts.ccrs_eval (one Isaac process
per unit, policy-only, deterministic, gradient-free, no optimizer/PBT/
curriculum writes — those invariants are already enforced and unit-tested in
the CCRS toolkit). This module only:

1. snapshots the live policy at a training-step boundary into an immutable
   read-only file (SnapshotSpec);
2. builds the ordered seed manifest (identity = manifest_id, part of every
   provenance record);
3. constructs the subprocess command lines / EvalRequest set for a given
   (snapshot, tau, manifest) triple;
4. loads per-seed results back (per-seed retained, never aggregate-only)
   as decision.SeedEvalResult.

No Isaac import here; the GPU work always happens in child processes so the
live trainer state cannot be touched by evaluation.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dexpbt_isaaclab.msa.decision import SeedEvalResult

# Pre-registered evaluation sizes. The PILOT size is the default wiring;
# FORMAL must be requested explicitly (never run by default).
SMOKE_NUM_SEEDS = 4
PILOT_NUM_SEEDS = 16
FORMAL_NUM_SEEDS = 256
SEED_MANIFEST_BASE = 310_000  # ordered manifest: BASE+1 .. BASE+N


def canonical_tau(tau: float) -> str:
    """Canonical tolerance representation used END TO END: the eval request
    CLI value, the raw-result match and SeedEvalResult.tau all derive from
    this exact string, so a tau_plus like 0.021956862... can never be
    written at one precision and matched at another (the old bug: written
    as f"{tau:.6f}" but matched with a 1e-9 tolerance)."""
    return format(float(tau), ".12g")


def build_seed_manifest(n_seeds: int = PILOT_NUM_SEEDS,
                        base: int = SEED_MANIFEST_BASE) -> tuple[str, list[int]]:
    """Deterministic ordered seed manifest and its identity string."""
    seeds = [base + i + 1 for i in range(n_seeds)]
    ident = hashlib.sha256(
        json.dumps(seeds, separators=(",", ":")).encode()).hexdigest()[:16]
    return f"msa_seeds_{base}_{n_seeds}_{ident}", seeds


@dataclass(frozen=True)
class SnapshotSpec:
    """An immutable read-only snapshot taken at a training-step boundary."""
    snapshot_id: str          # e.g. "w1_cycle3_pre"
    worker_id: int
    frame: int                # worker-local ledger frame at snapshot time
    path: str                 # frozen .pth copy (chmod a-w after write)

    @staticmethod
    def freeze(src_state_path: Path, snapshot_id: str, worker_id: int,
               frame: int, out_dir: Path) -> "SnapshotSpec":
        """Copy the checkpoint written at the step boundary and strip write
        permission so no later code path can mutate it."""
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / f"{snapshot_id}.pth"
        # byte copy without importing torch (works for any file)
        dst.write_bytes(Path(src_state_path).read_bytes())
        dst.chmod(dst.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        return SnapshotSpec(snapshot_id=snapshot_id, worker_id=worker_id,
                            frame=frame, path=str(dst))

    def assert_immutable(self) -> None:
        mode = os.stat(self.path).st_mode
        if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise PermissionError(f"snapshot {self.path} is writable; "
                                  "MSA snapshots must be read-only")


@dataclass
class EvalRequest:
    """One fixed-tolerance evaluation of one snapshot over the manifest.

    Expanded into len(seeds) ccrs_eval subprocess invocations (one Isaac
    process per unit, exactly like the CCRS sweep). eval_device is
    REQUIRED: evaluation must never silently share cuda:0 with a live
    trainer; the caller pins the eval GPU explicitly and build_env() maps
    it to CUDA_VISIBLE_DEVICES for the child process."""
    snapshot: SnapshotSpec
    tau: float
    manifest_id: str
    seeds: list[int]
    task: str
    run_id: str
    raw_dir: str
    eval_device: str = ""            # e.g. "cuda:7" or bare GPU index "7"
    num_envs: int = 512
    num_episodes: int = 512
    max_consecutive_successes: int = 50
    commands: list[list[str]] = field(default_factory=list)

    def gpu_index(self) -> str:
        dev = str(self.eval_device).strip()
        if not dev:
            raise ValueError(
                "EvalRequest.eval_device is required: refusing to default "
                "onto the trainers' cuda:0")
        return dev.split(":", 1)[1] if ":" in dev else dev

    def build_env(self, base_env: Optional[dict] = None) -> dict:
        """Child-process environment with the eval GPU pinned."""
        env = dict(base_env) if base_env is not None else dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = self.gpu_index()
        return env

    def build_commands(self, python_exe: str = "python") -> list[list[str]]:
        self.snapshot.assert_immutable()
        self.gpu_index()   # validate eval_device up front
        cmds = []
        for seed in self.seeds:
            cmds.append([
                python_exe, "-m", "dexpbt_isaaclab.scripts.ccrs_eval",
                "--task", self.task,
                "--run-id", self.run_id,
                "--worker-id", str(self.snapshot.worker_id),
                "--checkpoint-iteration", "0",
                "--checkpoint-frame", str(self.snapshot.frame),
                "--checkpoint-path", self.snapshot.path,
                "--fixed-tolerance", canonical_tau(self.tau),
                "--reset-seed", str(seed),
                "--num-envs", str(self.num_envs),
                "--num-episodes", str(self.num_episodes),
                "--deterministic", "true",
                "--max-consecutive-successes",
                str(self.max_consecutive_successes),
                "--evaluation-mode", "policy_only",
                "--raw-dir", self.raw_dir,
                "--headless",
            ])
        self.commands = cmds
        return cmds


def load_seed_results(raw_dir: Path, snapshot: SnapshotSpec, tau: float,
                      manifest_id: str, seeds: list[int],
                      metric: str = "normalized_nsucc") -> SeedEvalResult:
    """Collect per-seed results for (snapshot, tau) from ccrs_eval raw JSON.

    Every seed must be present with status=ok and the CANONICAL tolerance
    (format(x, '.12g') on both the request side and the stored raw value,
    so a tau written at full precision always matches its own request) —
    missing or mismatched cells fail closed (raise), they are never
    silently averaged over.
    """
    want_tau = canonical_tau(tau)
    per_seed: dict[int, float] = {}
    for f in Path(raw_dir).glob("*.json"):
        try:
            row = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if row.get("status") != "ok":
            continue
        if row.get("checkpoint_path") != snapshot.path:
            continue
        if canonical_tau(float(row.get("fixed_tolerance", -1))) != want_tau:
            continue
        per_seed[int(row["reset_seed"])] = float(row[metric])
    missing = [s for s in seeds if s not in per_seed]
    if missing:
        raise FileNotFoundError(
            f"snapshot {snapshot.snapshot_id} tau={want_tau}: missing "
            f"per-seed results for {len(missing)} seeds "
            f"(e.g. {missing[:5]}); fail closed")
    return SeedEvalResult(manifest_id=manifest_id,
                          tau=float(want_tau), seeds=list(seeds),
                          success=[per_seed[s] for s in seeds])
