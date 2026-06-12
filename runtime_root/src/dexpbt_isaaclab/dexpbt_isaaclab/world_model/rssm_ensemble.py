"""Full RSSM ensemble for stochastic EIR-MPPI scoring only.

This module hosts a lightweight manager over a list of *independent*
``RSSMLiteWorldModel`` members. The ensemble exists purely to provide an
epistemic-uncertainty (model-disagreement) signal to the H-step RSSM-MPPI
planner's sequence scoring. It does NOT:

* implement imagined posterior correction (prior-rollout only, as before);
* control the actor action (MPPI output is still only an ``SRC_RSSM`` EIR
  candidate that the reranker may or may not pick);
* alter the PPO loss path beyond training extra RSSM members;
* touch Elite Bank / Gated Residual / PBT Broadcast.

When the ensemble is disabled (default) the single-RSSM behaviour is
unchanged: nothing in this module runs unless ``--seq_wm_ensemble_enable``
and ``--eir_mppi_use_model_ensemble`` are set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from .seq_wm_base import SequenceBatch


@dataclass
class RSSMEnsembleConfig:
    enabled: bool = False
    size: int = 3
    train_all: bool = True
    # Reserved CLI hook; per-sample bootstrap masks are NOT implemented.
    # Diversity comes from independent init + independent sequence minibatches.
    use_bootstrap_masks: bool = True
    bootstrap_keep_prob: float = 0.8
    share_architecture: bool = True
    device: str = "cuda"

    # MPPI usage
    mppi_use_ensemble: bool = True
    mppi_max_models: int = 3
    mppi_model_subsample: bool = False

    # checkpoint / logging
    save_ensemble: bool = True
    load_mode: str = "reset"  # reset | resume

    def __post_init__(self) -> None:
        self.size = max(1, int(self.size))
        self.bootstrap_keep_prob = float(min(1.0, max(0.0, self.bootstrap_keep_prob)))
        self.mppi_max_models = max(1, int(self.mppi_max_models))
        if str(self.load_mode) not in ("reset", "resume"):
            self.load_mode = "reset"


class RSSMEnsembleManager:
    """Owns a list of independent RSSM world models for MPPI scoring.

    The members are full ``RSSMLiteWorldModel`` instances (no shared backbone).
    Member 0 is, by convention, the same model the rest of the pipeline already
    uses as the primary seq-WM (so single-model paths keep working). Members
    1..E-1 are extra independent models trained in parallel for disagreement.
    """

    def __init__(self, models: list[Any], cfg: RSSMEnsembleConfig) -> None:
        if not models:
            raise ValueError("RSSMEnsembleManager requires at least one model")
        self._models = list(models)
        self.cfg = cfg

    # ----- basic accessors -------------------------------------------------
    @property
    def models(self) -> list[Any]:
        return self._models

    @property
    def size(self) -> int:
        return len(self._models)

    def train(self) -> "RSSMEnsembleManager":
        for m in self._models:
            m.train()
        return self

    def eval(self) -> "RSSMEnsembleManager":
        for m in self._models:
            m.eval()
        return self

    def to(self, device: Any) -> "RSSMEnsembleManager":
        for m in self._models:
            m.to(device)
        return self

    # ----- checkpoint ------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "ensemble_size": int(self.size),
            "members": [m.state_dict() for m in self._models],
        }

    def load_state_dict(self, state: dict, strict: bool = False) -> int:
        """Load member state dicts. Returns the number of members loaded.

        Tolerant by design: if the checkpoint has fewer/more members than the
        live ensemble, only the overlapping prefix is loaded and a warning is
        printed; the run is never failed here.
        """
        members = state.get("members") if isinstance(state, dict) else None
        if not members:
            print("[RSSM-ensemble] load_state_dict: no 'members' in state; skipping")
            return 0
        n = min(len(members), len(self._models))
        if len(members) != len(self._models):
            print(
                f"[RSSM-ensemble] load_state_dict: checkpoint size {len(members)} != "
                f"live size {len(self._models)}; loading first {n} members"
            )
        loaded = 0
        for e in range(n):
            try:
                self._models[e].load_state_dict(members[e], strict=strict)
                loaded += 1
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[RSSM-ensemble] member {e} load failed: {exc}")
        return loaded

    def init_members_from_primary(self, *, member_start: int = 1) -> int:
        """Copy member-0 weights into members ``member_start..E-1``.

        Used when resuming a SeqWM checkpoint that lacks ``rssm_ensemble`` state
        so PBT restarts do not leave members 1..E-1 randomly initialized while
        member 0 matches the restored primary RSSM.
        """
        if self.size <= 1:
            return 0
        primary_sd = self._models[0].state_dict()
        copied = 0
        for e in range(max(1, int(member_start)), self.size):
            try:
                self._models[e].load_state_dict(primary_sd, strict=False)
                copied += 1
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[RSSM-ensemble] member {e} init_from_primary failed: {exc}")
        return copied

    # ----- inference -------------------------------------------------------
    @torch.no_grad()
    def encode_sequence_ensemble(
        self,
        batch: SequenceBatch,
        *,
        deterministic: bool = True,
    ) -> list[torch.Tensor]:
        """Return ``[belief_seq_e]`` for each member, each ``[T, N, D]``.

        Members that raise are skipped (list shorter than ``size``). The caller
        decides whether the remaining count is sufficient.
        """
        out: list[torch.Tensor] = []
        for e, m in enumerate(self._models):
            try:
                was_training = m.training
                m.eval()
                try:
                    belief_seq = m.encode_sequence(batch, deterministic=deterministic)
                finally:
                    if was_training:
                        m.train()
                out.append(belief_seq.detach())
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[RSSM-ensemble] member {e} encode_sequence failed: {exc}")
        return out


def make_bootstrap_generator(base_seed: int, member_index: int) -> torch.Generator:
    """Per-member CPU RNG so ensemble trainers sample independent minibatches."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(base_seed) * 1_000_003 + int(member_index) + 1)
    return g


def restore_rssm_ensemble_checkpoint(
    manager: RSSMEnsembleManager,
    payload: dict,
    *,
    wm_load_mode: str = "auto",
    ensemble_load_mode: str = "resume",
    primary_loaded: bool = True,
) -> dict[str, float]:
    """Restore ensemble member weights from a SeqWM checkpoint payload.

    Returns scalar metrics suitable for TB logging (e.g.
    ``init_from_primary``, ``ensemble_members_loaded``).

    Ensemble restore is attempted when ``ensemble_load_mode=='resume'`` or when
    a PBT restart passes ``wm_load_mode in ('auto', 'force')`` and the payload
    already contains ``rssm_ensemble`` (cross-copy / workspace checkpoint).
    """
    metrics: dict[str, float] = {
        "bootstrap_mask_active": 0.0,
    }
    if manager.size <= 1:
        return metrics

    wm_mode = str(wm_load_mode or "auto")
    ens_mode = str(ensemble_load_mode or "reset")
    if wm_mode == "reset":
        return metrics

    ens_state = payload.get("rssm_ensemble") if isinstance(payload, dict) else None
    should_restore = ens_mode == "resume" or (
        wm_mode in ("auto", "force") and isinstance(ens_state, dict)
    )
    if not should_restore:
        return metrics

    if isinstance(ens_state, dict):
        loaded = manager.load_state_dict(ens_state, strict=False)
        print(f"[RSSM-ensemble] loaded {loaded} member(s) from checkpoint")
        metrics["ensemble_members_loaded"] = float(loaded)
        return metrics

    if primary_loaded:
        copied = manager.init_members_from_primary()
        print(
            "[RSSM-ensemble] no ensemble state in checkpoint; "
            f"init_from_primary for {copied} member(s)"
        )
        metrics["init_from_primary"] = 1.0
    return metrics


__all__ = [
    "RSSMEnsembleConfig",
    "RSSMEnsembleManager",
    "make_bootstrap_generator",
    "restore_rssm_ensemble_checkpoint",
]
