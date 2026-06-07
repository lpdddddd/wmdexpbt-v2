"""
Bayesian posterior selector with Thompson Sampling.
Supports none strategy with fixed mean=0.
SharedPosteriorSelector: workspace-level shared file with atomic rename.

Phase A4 extension (2026-06-08): additive stage-aware posterior. The same
JSON file gains an optional ``__stage_posterior__`` key:

    {
        "none": {"n":..., "mean":..., "M2":..., "last_update_step":...},
        ..., "__pool__": [...],
        "__stage_posterior__": {
            "recovery":    {"<strategy>": {n, mean, M2, last_update_step}, ...},
            "early":       {...},
            "acquisition": {...},
            "mastery":     {...}
        }
    }

The schema is fully backward compatible: pre-A4 files load fine and the
stage_posterior simply starts empty. ``sample_stage(stage)`` does
PER-STRATEGY fallback to the global posterior whenever a particular
strategy lacks ``min_stage_n`` samples in that stage.
"""

import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# Stage name ordering kept in sync with stage_classifier.VALID_STAGES.
STAGES: tuple[str, ...] = ("recovery", "early", "acquisition", "mastery")


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
        # Stage-aware posterior. None of the entries are pre-allocated; the
        # first update for a (stage, strategy) pair instantiates the state
        # at n=1, mean=0.0, M2=1.0 (matching the global init).
        self.stage_posterior: dict[str, dict[str, PosteriorState]] = {
            stage: {} for stage in STAGES
        }
        # Per-(stage, strategy) fallback counter; reset on save/load only.
        self.fallback_counts: dict[tuple[str, str], int] = {}

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

    # ----- stage-aware extension (A4) -----

    def _stage_state(self, stage: str, strategy: str) -> PosteriorState:
        """Lazily allocate the (stage, strategy) posterior entry."""
        if stage not in self.stage_posterior:
            self.stage_posterior[stage] = {}
        if strategy not in self.stage_posterior[stage]:
            # None must always start with mean=0; others also start at 0
            # so the very first update is unbiased.
            self.stage_posterior[stage][strategy] = PosteriorState(
                n=1, mean=0.0, M2=1.0,
            )
        return self.stage_posterior[stage][strategy]

    def update_batch_stage(
        self,
        delta_w: dict[str, float],
        stage: str,
        step: int,
        weights: Optional[dict[str, float]] = None,
        sample_count_weighted: bool = True,
    ) -> None:
        """Update both the global posterior AND the per-stage posterior.

        ``stage`` must be one of ``STAGES`` or this is a no-op for the
        stage path (the global path still updates).
        """
        # Always update global first (preserves prior behaviour).
        self.update_batch(
            delta_w, step, weights=weights,
            sample_count_weighted=sample_count_weighted,
        )
        if stage not in STAGES:
            return
        weights = dict(weights or {})
        for strategy, delta in delta_w.items():
            if strategy not in self.posterior:
                continue
            w = float(weights.get(strategy, 1.0))
            if w <= 0.0:
                continue
            ps = self._stage_state(stage, strategy)
            if strategy == "none":
                ps.n += 1 if sample_count_weighted else 1
                ps.M2 = ps.M2 + delta * delta
                ps.mean = 0.0
                ps.last_update_step = step
                continue
            mean_old = ps.mean
            if sample_count_weighted and 0.0 < w < 1.0:
                ps.n = max(1, int(ps.n + w))
            else:
                ps.n += 1
            delta_mean = delta - mean_old
            ps.mean = mean_old + delta_mean / ps.n
            ps.M2 = ps.M2 + delta_mean * (delta - ps.mean) * w
            ps.last_update_step = step

    def sample_stage(
        self,
        stage: str,
        *,
        context: str = "stage",
        min_stage_n: int = 3,
    ) -> str:
        """Stage-aware Thompson sampling with PER-STRATEGY fallback.

        For each non-``none`` strategy s:
          - if ``context == 'stage'`` AND
            ``stage_posterior[stage][s].n >= min_stage_n``  -> use stage
          - else -> use global posterior (record fallback)
        ``none`` is always sampled at 0.0 (mean fixed).
        """
        if context not in ("stage", "global"):
            raise ValueError(
                f"context must be 'stage' or 'global'; got {context!r}"
            )
        if context == "global" or stage not in STAGES:
            return self.sample()
        samples: dict[str, float] = {}
        for s in self.strategies:
            if s == "none":
                samples["none"] = 0.0
                continue
            stage_state = self.stage_posterior.get(stage, {}).get(s)
            if stage_state is not None and stage_state.n >= min_stage_n:
                samples[s] = random.gauss(stage_state.mean, stage_state.std)
            else:
                # Per-strategy fallback to global.
                self.fallback_counts[(stage, s)] = (
                    self.fallback_counts.get((stage, s), 0) + 1
                )
                p = self.posterior[s]
                samples[s] = random.gauss(p.mean, p.std)
        return max(samples, key=samples.get)

    # ----- (de)serialisation -----

    def to_dict(self) -> dict:
        out = {s: asdict(p) for s, p in self.posterior.items()}
        # Only emit __stage_posterior__ when at least one stage has data,
        # to keep on-disk JSON minimal for legacy / non-stage runs.
        any_stage = any(self.stage_posterior.get(st) for st in STAGES)
        if any_stage:
            out["__stage_posterior__"] = {
                stage: {
                    strat: asdict(state)
                    for strat, state in self.stage_posterior.get(stage, {}).items()
                }
                for stage in STAGES
            }
        return out

    def load_state(self, state: dict) -> None:
        for s, d in state.items():
            # Track C (2026-06-06): skip the pool metadata key written by
            # SharedPosteriorSelector.save(); not a strategy entry.
            if s == "__pool__":
                continue
            if s == "__stage_posterior__":
                # Phase A4: stage-aware additive payload. Tolerate missing
                # stages and unknown strategy names (drop silently).
                if not isinstance(d, dict):
                    continue
                for stage_name, stage_dict in d.items():
                    if stage_name not in STAGES or not isinstance(stage_dict, dict):
                        continue
                    bucket = self.stage_posterior.setdefault(stage_name, {})
                    for strat, sdict in stage_dict.items():
                        if strat not in self.posterior:
                            continue
                        try:
                            ps = PosteriorState(**sdict)
                        except TypeError:
                            continue
                        if strat == "none":
                            ps.mean = 0.0
                        bucket[strat] = ps
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

    def update_and_save_stage(
        self,
        delta_w: dict[str, float],
        stage: str,
        step: int,
        weights: Optional[dict[str, float]] = None,
    ) -> None:
        """A4: stage-aware update + atomic save.

        Updates the global posterior AND the per-stage posterior, then
        persists the full schema (including ``__stage_posterior__``).
        Use this from the PBT observer when ``--wm_selector_context=stage``.
        """
        self.update_batch_stage(delta_w, stage, step, weights=weights)
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