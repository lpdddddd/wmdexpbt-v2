"""
Bayesian posterior selector with Thompson Sampling.
Supports none strategy with fixed mean=0.
SharedPosteriorSelector: workspace-level shared file with atomic rename.
"""

import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class PosteriorState:
    """Welford posterior state for a single strategy."""
    n: int = 1
    mean: float = 0.0
    M2: float = 1.0   # sum of squared deviations
    last_update_step: int = 0

    @property
    def std(self) -> float:
        return math.sqrt(max(self.M2 / max(self.n, 1), 1e-6))


class BayesianPosteriorSelector:
    """
    In-process Bayesian posterior selector with Thompson Sampling.
    Non-shared version — state lives in memory.
    """

    STRATEGIES = ["none", "aux_low", "aux_mid", "value_imag_h1", "value_imag_h3"]

    def __init__(self, strategies: list[str] | None = None,
                 none_fixed_mean: bool = True):
        self.strategies = strategies or list(self.STRATEGIES)
        self.none_fixed_mean = none_fixed_mean
        self.posterior = {
            s: PosteriorState(n=1, mean=0.0, M2=1.0)
            for s in self.strategies
        }

    def update_batch(
        self,
        delta_w: dict[str, float],
        step: int,
        weights: Optional[dict[str, float]] = None,
        sample_count_weighted: bool = True,
    ) -> None:
        """
        Update posterior using Welford's online algorithm.

        delta_w: {strategy_name: r_meta_delta_vs_none}.

        Patch 4 (rev 2, 2026-05-30): supports per-strategy `weights` so the
        CF runner can downweight low_signal blocks (default behaviour: skip
        entirely by setting weight=0). For Welford with weight w in [0, 1]:

          * weight == 0  -> the strategy is skipped (no update to n/mean/M2).
          * weight in (0,1) -> effective sample count is w; the increment
            in n is `w` (truncated to int when `sample_count_weighted` is
            False) and the Welford update step uses fractional n.

        The simpler, more robust path -- and the one used by the CF
        runner today -- is `skip` (weight=0) for low_signal blocks.
        """
        weights = dict(weights or {})
        for strategy, delta in delta_w.items():
            if strategy not in self.posterior:
                continue

            w = float(weights.get(strategy, 1.0))
            if w <= 0.0:
                # Patch 4 (rev 2): explicit skip path so caller intent is
                # auditable from log output.
                continue

            p = self.posterior[strategy]

            if strategy == "none":
                # none mean stays fixed at 0, only update uncertainty
                p.n += 1 if sample_count_weighted else 1
                p.M2 = p.M2 + delta * delta
                p.mean = 0.0
                p.last_update_step = step
                continue

            # Welford update for non-none strategies; supports fractional
            # `weight` via effective sample increment.
            mean_old = p.mean
            if sample_count_weighted and 0.0 < w < 1.0:
                p.n = max(1, int(p.n + w))
            else:
                p.n += 1
            delta_mean = delta - mean_old
            p.mean = mean_old + delta_mean / p.n
            p.M2 = p.M2 + delta_mean * (delta - p.mean) * w
            p.last_update_step = step

    def sample(self) -> str:
        """
        Thompson Sampling: sample from N(μ_w, √(M2_w/n_w)) for each strategy.
        none returns fixed 0.0 (not sampled).
        Returns the strategy with highest sample.
        """
        samples = {}
        for s in self.strategies:
            if s == "none":
                samples["none"] = 0.0
                continue
            p = self.posterior[s]
            std = p.std
            samples[s] = random.gauss(p.mean, std)

        return max(samples, key=samples.get)

    def to_dict(self) -> dict:
        return {
            s: asdict(p) for s, p in self.posterior.items()
        }

    def load_state(self, state: dict) -> None:
        for s, d in state.items():
            # Track C (2026-06-06): skip the pool metadata key written by
            # SharedPosteriorSelector.save(); not a strategy entry.
            if s == "__pool__":
                continue
            if s in self.posterior:
                loaded = PosteriorState(**d)
                if s == "none":
                    loaded.mean = 0.0
                self.posterior[s] = loaded


class SharedPosteriorSelector(BayesianPosteriorSelector):
    """
    Workspace-level shared posterior selector.
    Reads/writes wm_posterior_latest.json with atomic rename.

    Only policy_idx=0 or CF script writes; other workers only read.
    """

    def __init__(self, shared_path: Path | str,
                 strategies: list[str] | None = None,
                 none_fixed_mean: bool = True):
        super().__init__(
            strategies=strategies or BayesianPosteriorSelector.STRATEGIES,
            none_fixed_mean=none_fixed_mean,
        )
        self.shared_path = Path(shared_path)

    def load_or_init(self, policy_idx: int = 0) -> None:
        """Load from shared file if exists; otherwise init default and write if policy_idx==0."""
        if self.shared_path.exists():
            try:
                state = json.loads(self.shared_path.read_text())
                self.load_state(state)
                print(f"[Posterior] Loaded shared posterior from {self.shared_path}")
                return
            except Exception as e:
                print(f"[Posterior] Failed to load shared posterior ({e}), re-initializing")

        # Init default
        self.posterior = {
            s: PosteriorState(n=1, mean=0.0, M2=1.0)
            for s in self.strategies
        }
        if policy_idx == 0:
            self.save()

    def save(self) -> None:
        """Atomic rename write with fsync.

        Track C (2026-06-06): also persist the active strategy pool under
        a reserved `__pool__` key so post-hoc reports / acceptance checks
        can verify h3 (and aux_low) are not present.
        """
        state = self.to_dict()
        state["__pool__"] = list(self.strategies)
        self._atomic_write_json(self.shared_path, state)

    def update_and_save(
        self,
        delta_w: dict[str, float],
        step: int,
        weights: Optional[dict[str, float]] = None,
    ) -> None:
        """Update posterior + atomic save.

        Patch 4 (rev 2, 2026-05-30): forwards optional per-strategy weights
        so the CF block runner's `posterior_weights` (with values 0/0.25/1)
        propagate end-to-end.
        """
        self.update_batch(delta_w, step, weights=weights)
        self.save()

    @staticmethod
    def _atomic_write_json(path: Path, data: dict) -> None:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, indent=2))
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        tmp.replace(path)