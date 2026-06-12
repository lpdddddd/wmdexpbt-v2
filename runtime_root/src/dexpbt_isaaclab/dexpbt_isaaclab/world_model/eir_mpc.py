"""EIR-MPC orchestration: elite candidate retrieval + reranking.

This module composes the kept ERD action scorer with the Elite Recovery Bank to
build a strong candidate set (actor mean + Gaussian local + elite-local +
elite-archive + optional RSSM-reranked) and selects the best recovery action by
a recovery-aware score. It deliberately does NOT pull the main actor ``mu`` --
the selected best action is consumed by the gated recovery residual head
(``recovery_residual.py``). ERD's raw-mu distill path is disabled whenever EIR is
enabled.

Candidate tensors keep shape ``[B_risky, K_total, action_dim]`` and carry
per-candidate provenance so we can verify whether the *best* action actually
comes from elite / archive memory rather than Gaussian noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn.functional as F

from .elite_recovery_distill import (
    ERDConfig,
    _build_risky_mask,
    _combined_score_from_logits,
    _get_a2c_network,
    _masked_bce,
    _read_minibatch_range,
)
from .recovery_residual import (
    RecoveryResidualBridge,
    compute_recovery_residual_loss,
)

# EIR distill block reasons (parallel to ERD's).
EIR_OK = 0
EIR_DISABLED = 1
EIR_NO_CACHE = 2
EIR_NO_CAPTURE = 3
EIR_WARMUP_FRAMES = 4
EIR_WARMUP_ITERS = 5
EIR_LOW_VALID_FRACTION = 6
EIR_NO_RISKY = 7
EIR_NOT_ELIGIBLE = 8
EIR_TOL_ABOVE_START = 9


# Candidate provenance ids.
SRC_ACTOR = 0
SRC_GAUSSIAN = 1
SRC_ELITE_LOCAL = 2
SRC_ELITE_ARCHIVE = 3
SRC_RSSM = 4
_SRC_NAMES = {
    SRC_ACTOR: "actor",
    SRC_GAUSSIAN: "gaussian",
    SRC_ELITE_LOCAL: "elite",
    SRC_ELITE_ARCHIVE: "archive",
    SRC_RSSM: "rssm",
}


@dataclass
class EIRConfig:
    enabled: bool = False
    horizons: tuple[int, ...] = (4, 8)
    action_dim: int = 23
    feature_dim: int = 256

    # candidate budget
    local_candidate_k: int = 16
    elite_candidate_k: int = 32
    archive_candidate_k: int = 16
    action_noise_std: float = 0.15
    max_action_delta: float = 0.5
    noise_cap_by_tol: bool = True

    # reranking score weights
    score_success_weight: float = 1.0
    score_improve_weight: float = 0.7
    score_recovery_weight: float = 0.5
    score_drop_weight: float = 1.5
    elite_priority_weight: float = 0.2
    elite_similarity_weight: float = 0.1
    action_delta_weight: float = 0.05
    clip_penalty_weight: float = 0.1

    # gates
    min_score_margin: float = 0.02
    min_elite_margin: float = 0.01
    max_drop_prob_for_teacher: float = 0.35

    # staging tolerances (reuse ERD semantics)
    start_tol: float = 0.04
    floor_tol: float = 0.02
    hard_floor_tol: float = 0.012
    near_success_mult: float = 2.0
    only_lifted: bool = True

    scorer_loss_weight: float = 0.01
    start_after_frames: int = 200_000_000
    scorer_warmup_iters: int = 2000
    min_target_valid_fraction: float = 0.2
    min_risky_fraction: float = 0.02

    def __post_init__(self) -> None:
        self.horizons = tuple(int(h) for h in self.horizons if int(h) > 0) or (4, 8)

    def distill_lambda_for_tol(self, tol: float) -> float:
        # staging gate identical in spirit to ERD: residual distill only once tol
        # has dropped into the floor band.
        if not self.enabled:
            return 0.0
        return 1.0 if float(tol) <= float(self.start_tol) else 0.0


def _noise_std_for_tol(std: float, cap_by_tol: bool, floor_tol: float, hard_tol: float, tol_mean: float) -> float:
    if not cap_by_tol:
        return std
    if tol_mean <= hard_tol:
        return std * 0.5
    if tol_mean <= floor_tol:
        return std * 0.75
    return std


def generate_eir_candidates(
    mu_r: torch.Tensor,
    *,
    cfg: EIRConfig,
    bank: Any = None,
    archive_bank: Any = None,
    feat_r: Optional[torch.Tensor] = None,
    tol_mean: float = 1.0,
    stage: Optional[int] = None,
    rssm_actions: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Build the EIR candidate set around the clipped actor mean.

    Returns dict with ``cand[B,K,A]``, ``source_id[B,K]`` (long), ``priority[B,K]``,
    ``similarity[B,K]``. Candidate 0 is always the clipped actor mean.
    """
    B, A = mu_r.shape
    dev, dtype = mu_r.device, mu_r.dtype
    base = mu_r.detach().clamp(-1.0, 1.0)

    cand_blocks: list[torch.Tensor] = []
    src_blocks: list[torch.Tensor] = []
    pri_blocks: list[torch.Tensor] = []
    sim_blocks: list[torch.Tensor] = []

    def _add(actions: torch.Tensor, src_id: int, pri: torch.Tensor, sim: torch.Tensor) -> None:
        cand_blocks.append(actions)
        k = actions.shape[1]
        src_blocks.append(torch.full((B, k), int(src_id), device=dev, dtype=torch.long))
        pri_blocks.append(pri)
        sim_blocks.append(sim)

    zeros_k = lambda k: torch.zeros(B, k, device=dev, dtype=dtype)  # noqa: E731

    # actor mean
    _add(base[:, None, :], SRC_ACTOR, zeros_k(1), zeros_k(1))

    # gaussian local
    kg = max(0, int(cfg.local_candidate_k))
    if kg > 0:
        std = _noise_std_for_tol(
            float(cfg.action_noise_std), bool(cfg.noise_cap_by_tol),
            float(cfg.floor_tol), float(cfg.hard_floor_tol), float(tol_mean),
        )
        noise = torch.randn(B, kg, A, device=dev, dtype=dtype) * std
        noise = noise.clamp(-float(cfg.max_action_delta), float(cfg.max_action_delta))
        gauss = (base[:, None, :] + noise).clamp(-1.0, 1.0)
        _add(gauss, SRC_GAUSSIAN, zeros_k(kg), zeros_k(kg))

    # elite local
    ke = max(0, int(cfg.elite_candidate_k))
    if ke > 0 and bank is not None and feat_r is not None and getattr(bank, "size", 0) > 0:
        acts, sim, pri, valid = bank.retrieve_candidates(feat_r, k=ke, stage=stage)
        if valid:
            _add(acts.to(dtype), SRC_ELITE_LOCAL, pri.to(dtype), sim.to(dtype))

    # elite archive
    ka = max(0, int(cfg.archive_candidate_k))
    if ka > 0 and archive_bank is not None and feat_r is not None and getattr(archive_bank, "size", 0) > 0:
        acts, sim, pri, valid = archive_bank.retrieve_candidates(feat_r, k=ka, stage=stage)
        if valid:
            _add(acts.to(dtype), SRC_ELITE_ARCHIVE, pri.to(dtype), sim.to(dtype))

    # rssm reranked candidate(s) - actions supplied by planner
    if rssm_actions is not None and rssm_actions.numel() > 0:
        ra = rssm_actions.to(device=dev, dtype=dtype)
        if ra.dim() == 2:
            ra = ra[:, None, :]
        kr = ra.shape[1]
        _add(ra.clamp(-1.0, 1.0), SRC_RSSM, zeros_k(kr), zeros_k(kr))

    cand = torch.cat(cand_blocks, dim=1)
    source_id = torch.cat(src_blocks, dim=1)
    priority = torch.cat(pri_blocks, dim=1)
    similarity = torch.cat(sim_blocks, dim=1)
    return {
        "cand": cand,
        "source_id": source_id,
        "priority": priority,
        "similarity": similarity,
    }


def _boundary_penalty(action: torch.Tensor) -> torch.Tensor:
    # squared overshoot beyond [-1,1] of the pre-clamp candidate; for clamped
    # candidates this is ~0, but kept for parity / future raw candidates.
    over = (action.abs() - 1.0).clamp_min(0.0)
    return (over ** 2).sum(dim=-1)


def rerank_candidates(
    *,
    cand: torch.Tensor,
    source_id: torch.Tensor,
    priority: torch.Tensor,
    similarity: torch.Tensor,
    scorer: Any,
    feat_r: torch.Tensor,
    cfg: EIRConfig,
    erd_cfg: ERDConfig,
    recovery_probs: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """Score candidates and select the best recovery action per row.

    Returns best_action[B,A], best_idx[B], best_source[B], margin[B],
    plus aggregate source / best-source fractions and prob summaries.
    """
    B, K, A = cand.shape
    dev = cand.device
    feat_rep = feat_r.detach()[:, None, :].expand(B, K, feat_r.shape[-1]).reshape(B * K, -1)
    cand_flat = cand.reshape(B * K, A)
    with torch.no_grad():
        logits = scorer(feat_rep, cand_flat)
        score, succ, impr, drop = _combined_score_from_logits(erd_cfg, logits)
        score = score.view(B, K)
        succ = succ.view(B, K)
        impr = impr.view(B, K)
        drop = drop.view(B, K)

        # recovery term (optional, from RSSM head rescoring) broadcast if given
        rec = torch.zeros(B, K, device=dev, dtype=score.dtype)
        if recovery_probs is not None:
            rp = recovery_probs.to(device=dev, dtype=score.dtype)
            if rp.dim() == 1:
                rp = rp[:, None].expand(B, K)
            rec = rp.view(B, K)

        # priority normalization to ~[0,1] for the bonus
        pr = priority
        pr_min = pr.min(dim=1, keepdim=True).values
        pr_max = pr.max(dim=1, keepdim=True).values
        pr_norm = (pr - pr_min) / (pr_max - pr_min + 1e-6)

        base = cand[:, 0:1, :]  # clipped actor mean
        action_delta = (cand - base).norm(dim=-1)  # [B,K]
        clip_pen = _boundary_penalty(cand).view(B, K)

        total = (
            cfg.score_success_weight * succ
            + cfg.score_improve_weight * impr
            + cfg.score_recovery_weight * rec
            - cfg.score_drop_weight * drop
            + cfg.elite_priority_weight * pr_norm
            + cfg.elite_similarity_weight * similarity
            - cfg.action_delta_weight * (action_delta ** 2)
            - cfg.clip_penalty_weight * clip_pen
        )

        best_idx = total.argmax(dim=1)
        rows = torch.arange(B, device=dev)
        best_action = cand[rows, best_idx]
        best_score = total[rows, best_idx]
        actor_score = total[:, 0]
        margin = best_score - actor_score
        best_drop = drop[rows, best_idx]
        best_source = source_id[rows, best_idx]

        # eligibility: stronger margin + low drop; elite/archive need elite margin
        is_elite_best = (best_source == SRC_ELITE_LOCAL) | (best_source == SRC_ELITE_ARCHIVE)
        margin_gate = torch.where(
            is_elite_best,
            margin > float(cfg.min_elite_margin),
            margin > float(cfg.min_score_margin),
        )
        eligible = margin_gate & (best_drop <= float(cfg.max_drop_prob_for_teacher))

        # source fractions
        def frac(mask: torch.Tensor) -> float:
            denom = float(B * K)
            return float(mask.float().sum().item() / denom) if denom > 0 else 0.0

        def best_frac(src_id: int) -> float:
            return float((best_source == src_id).float().mean().item()) if B > 0 else 0.0

        out = {
            "best_action": best_action,
            "best_idx": best_idx,
            "best_source": best_source,
            "best_score": best_score,
            "actor_score": actor_score,
            "margin": margin,
            "best_drop": best_drop,
            "eligible": eligible,
            "best_succ": succ[rows, best_idx],
            "best_impr": impr[rows, best_idx],
            "actor_succ": succ[:, 0],
            "actor_drop": drop[:, 0],
            "candidate_source_frac": {
                _SRC_NAMES[s]: frac(source_id == s) for s in _SRC_NAMES
            },
            "best_source_frac": {
                _SRC_NAMES[s]: best_frac(s) for s in _SRC_NAMES
            },
            "elite_similarity_mean": float(
                similarity[(source_id == SRC_ELITE_LOCAL) | (source_id == SRC_ELITE_ARCHIVE)].mean().item()
            ) if bool(((source_id == SRC_ELITE_LOCAL) | (source_id == SRC_ELITE_ARCHIVE)).any()) else 0.0,
            "elite_action_delta_norm": float(
                action_delta[(source_id == SRC_ELITE_LOCAL) | (source_id == SRC_ELITE_ARCHIVE)].mean().item()
            ) if bool(((source_id == SRC_ELITE_LOCAL) | (source_id == SRC_ELITE_ARCHIVE)).any()) else 0.0,
            "margin_mean": float(margin.mean().item()),
            "best_score_mean": float(best_score.mean().item()),
            "actor_score_mean": float(actor_score.mean().item()),
        }
    return out


@dataclass
class EIRBridge:
    """Holds all EIR runtime state for the PPO aux-loss hook.

    Reuses the ERD action scorer + target cache + actor_mlp feature capture, but
    routes the actor-improving signal through the recovery residual head (never a
    raw-mu distill) and harvests the elite bank.
    """

    eir_cfg: EIRConfig
    erd_cfg: ERDConfig
    scorer: Any = None
    residual_bridge: Optional[RecoveryResidualBridge] = None
    bank: Any = None
    archive_bank: Any = None
    rssm_reranker: Any = None
    capture: Any = None  # ERDCapture (actor_mlp forward output)
    agent_ref: Any = None
    cache: Any = None
    broadcast_cfg: Any = None  # EliteBroadcastConfig (PBT elite broadcast, Phase D)
    # Light RSSM reranking hooks (Phase E wire-in). ``rssm_belief_provider`` maps
    # the risky minibatch features -> a fixed belief [B_risky, D] (or None when a
    # per-row belief is unavailable, in which case the reranker self-disables and
    # the candidate pool is unaffected). ``rssm_predict_fn`` is the WM
    # ``heads.predict_transition``; ``rssm_value_fn`` is ``heads.predict_value``.
    rssm_belief_provider: Any = None
    rssm_predict_fn: Any = None
    rssm_value_fn: Any = None
    last_rssm_candidate_added: float = 0.0
    # Full H-step MPPI hooks (Phase: full MPPI). ``rssm_planner`` is an
    # RSSMMPPIPlanner; ``rssm_model`` is the seq WM model exposing
    # ``get_state_from_belief`` / ``imagine_step``; ``rssm_belief_cache`` is the
    # per-rollout RSSMBeliefCache (built once in train.py). When present and
    # active these supersede the light reranker for producing the SRC_RSSM
    # candidate. All optional: missing -> light path -> elite/scorer fallback.
    rssm_planner: Any = None
    rssm_model: Any = None
    # Optional list of independent RSSM models for the ensemble MPPI scoring
    # path. None -> single-model MPPI (unchanged). Set only when
    # --eir_mppi_use_model_ensemble is enabled.
    rssm_models: Any = None
    rssm_belief_cache: Any = None
    last_mppi_active: float = 0.0
    last_mppi_belief_valid: float = 0.0
    last_mppi_use_model_ensemble: float = 0.0
    last_mppi_ensemble_valid_size: float = 0.0
    last_mppi_ensemble_score_std: float = 0.0
    last_mppi_ensemble_configured: float = 0.0
    last_mppi_ensemble_available_size: float = 0.0
    last_mppi_ensemble_used: float = 0.0
    last_mppi_ensemble_fallback_reason: float = 0.0
    last_mppi_score_std: float = 0.0
    last_mppi_drop_cvar: float = 0.0
    last_mppi_uncertainty_cvar: float = 0.0
    last_mppi_risk_adjusted_margin: float = 0.0
    last_mppi_stochastic_rollouts: float = 0.0
    last_mppi_num_latent_samples: float = 0.0

    scorer_warmup_iters_done: int = 0

    # logged metrics (mirrored to eir/* TB tags)
    last_active: float = 0.0
    last_scorer_active: float = 0.0
    last_residual_active: float = 0.0
    last_block_reason: int = EIR_DISABLED
    last_scorer_loss: float = 0.0
    last_total_loss: float = 0.0
    last_risky_fraction: float = 0.0
    last_target_valid_fraction: float = 0.0
    last_score_margin: float = 0.0
    last_best_score: float = 0.0
    last_actor_score: float = 0.0
    last_candidate_source_frac: dict = field(default_factory=dict)
    last_best_source_frac: dict = field(default_factory=dict)
    last_elite_similarity_mean: float = 0.0
    last_elite_action_delta_norm: float = 0.0
    last_elite_bank_size: float = 0.0
    last_bank_metrics: dict = field(default_factory=dict)
    last_bank_feature_source: int = 0
    bank_feature_update_count: int = 0
    bank_feature_source_actor_count: int = 0
    bank_feature_source_fallback_count: int = 0

    def record_bank_feature_source(self, source_flag: int) -> None:
        """Track whether bank updates use actor_mlp (0) or fallback keys (1)."""
        self.bank_feature_update_count += 1
        if int(source_flag) == 0:
            self.bank_feature_source_actor_count += 1
        else:
            self.bank_feature_source_fallback_count += 1
        self.last_bank_feature_source = int(source_flag)

    @property
    def bank_feature_source_actor_fraction(self) -> float:
        n = int(self.bank_feature_update_count)
        return float(self.bank_feature_source_actor_count) / float(n) if n > 0 else 0.0

    @property
    def bank_feature_source_fallback_fraction(self) -> float:
        n = int(self.bank_feature_update_count)
        return float(self.bank_feature_source_fallback_count) / float(n) if n > 0 else 0.0

    def reset_per_call(self) -> None:
        self.last_active = 0.0
        self.last_scorer_active = 0.0
        self.last_residual_active = 0.0
        self.last_block_reason = EIR_OK
        self.last_scorer_loss = 0.0
        self.last_total_loss = 0.0
        self.last_risky_fraction = 0.0
        self.last_score_margin = 0.0
        self.last_best_score = 0.0
        self.last_actor_score = 0.0
        self.last_candidate_source_frac = {}
        self.last_best_source_frac = {}
        self.last_elite_similarity_mean = 0.0
        self.last_elite_action_delta_norm = 0.0


def _stage_for_tol(eir_cfg: EIRConfig, tol_mean: float) -> Optional[int]:
    if tol_mean <= float(eir_cfg.hard_floor_tol):
        return 2
    if tol_mean <= float(eir_cfg.floor_tol):
        return 1
    return None


def _maybe_rssm_candidate_action(
    bridge: EIRBridge,
    *,
    feat_r: torch.Tensor,
    mu_r: torch.Tensor,
    cand_out: dict[str, torch.Tensor],
    start: int,
    end: int,
    risky: torch.Tensor,
    frames: int,
) -> Optional[torch.Tensor]:
    """Produce the SRC_RSSM candidate first-action for the risky rows.

    Tries full H-step MPPI first (planner + model + per-rollout belief cache),
    then falls back to the light single-step reranker. Returns ``best_action
    [B_risky, A]`` when a model-preferred action is available and active, else
    ``None`` (candidate pool unchanged).
    """
    dev = cand_out["cand"].device
    dtype = cand_out["cand"].dtype
    mu_clamped = mu_r.detach().clamp(-1.0, 1.0)

    # ---- belief for risky rows from the per-rollout cache ----
    belief_r: Optional[torch.Tensor] = None
    belief_r_ensemble: Optional[list] = None
    cache_obj = bridge.rssm_belief_cache
    if cache_obj is not None and getattr(cache_obj, "valid", False):
        bridge.last_mppi_belief_valid = 1.0
        sl = cache_obj.slice_rows(int(start), int(end))
        if sl is not None and int(sl.shape[0]) == int(risky.shape[0]):
            belief_r = sl[risky].to(device=dev, dtype=dtype)
        # ensemble per-member beliefs for the risky rows (None entries kept)
        if bridge.rssm_models is not None and hasattr(cache_obj, "slice_rows_ensemble"):
            sl_ens = cache_obj.slice_rows_ensemble(int(start), int(end))
            if sl_ens is not None:
                tmp_list: list = []
                for b in sl_ens:
                    if b is None or int(b.shape[0]) != int(risky.shape[0]):
                        tmp_list.append(None)
                    else:
                        tmp_list.append(b[risky].to(device=dev, dtype=dtype))
                if any(x is not None for x in tmp_list):
                    belief_r_ensemble = tmp_list
    # legacy provider (returns belief directly from features) as a secondary
    if belief_r is None and bridge.rssm_belief_provider is not None:
        try:
            belief_r = bridge.rssm_belief_provider(feat_r)
        except Exception:
            belief_r = None

    # ---- FULL H-step MPPI ----
    planner = bridge.rssm_planner
    if planner is not None and bridge.rssm_model is not None:
        try:
            out = planner.plan(
                belief_flat=belief_r,
                actor_action=mu_clamped,
                candidate_actions=cand_out["cand"].detach(),
                rssm_model=bridge.rssm_model,
                frames=int(frames),
                belief_flat_ensemble=belief_r_ensemble,
                rssm_models=bridge.rssm_models,
            )
            bridge.last_mppi_active = 1.0 if out.get("active") else 0.0
            bridge.last_mppi_use_model_ensemble = float(
                getattr(planner, "last_use_model_ensemble", 0.0)
            )
            bridge.last_mppi_ensemble_valid_size = float(
                getattr(planner, "last_ensemble_valid_size", 0.0)
            )
            bridge.last_mppi_ensemble_score_std = float(
                getattr(planner, "last_ensemble_score_std", 0.0)
            )
            bridge.last_mppi_ensemble_configured = float(
                getattr(planner, "last_ensemble_configured", 0.0)
            )
            bridge.last_mppi_ensemble_available_size = float(
                getattr(planner, "last_ensemble_available_size", 0.0)
            )
            bridge.last_mppi_ensemble_used = float(
                getattr(planner, "last_ensemble_used", 0.0)
            )
            bridge.last_mppi_ensemble_fallback_reason = float(
                getattr(planner, "last_ensemble_fallback_reason", 0.0)
            )
            bridge.last_mppi_stochastic_rollouts = float(
                getattr(planner, "last_stochastic_rollouts", 0.0)
            )
            bridge.last_mppi_num_latent_samples = float(
                getattr(planner, "last_num_latent_samples", 0.0)
            )
            bridge.last_mppi_score_std = float(getattr(planner, "last_score_std", 0.0))
            bridge.last_mppi_drop_cvar = float(getattr(planner, "last_drop_cvar", 0.0))
            bridge.last_mppi_uncertainty_cvar = float(
                getattr(planner, "last_uncertainty_cvar", 0.0)
            )
            bridge.last_mppi_risk_adjusted_margin = float(
                out.get("risk_adjusted_score_margin", torch.tensor(0.0)).mean().item()
                if isinstance(out.get("risk_adjusted_score_margin"), torch.Tensor)
                else getattr(planner, "last_risk_adjusted_margin", 0.0)
            )
            if out.get("active") and out.get("best_action") is not None:
                return out["best_action"]
            # planner ran but inactive (low margin / uncertain / disabled): do
            # NOT fall through to light if planner is the configured path.
            return None
        except Exception as exc:
            print(f"[EIR-MPPI] planner failed, falling back to light: {exc}")

    # ---- LIGHT single-step reranking fallback ----
    rk = bridge.rssm_reranker
    if rk is not None and bridge.rssm_predict_fn is not None:
        rk_out = rk.score_candidates(
            belief_flat=belief_r,
            cand_actions=cand_out["cand"],
            predict_fn=bridge.rssm_predict_fn,
            value_fn=bridge.rssm_value_fn,
            frames=int(frames),
        )
        if rk_out.get("active") and rk_out.get("best_action") is not None:
            return rk_out["best_action"]
    return None


def compute_eir_loss(bridge: EIRBridge, frames: int = 0) -> dict[str, torch.Tensor]:
    """EIR aux loss: ERD-style scorer BCE + gated recovery residual distill.

    The main actor ``mu`` receives NO direct gradient from this loss (constraint
    19.1): candidate features are detached for both the scorer and the residual
    head, and ``mu_base`` is detached inside the residual loss.
    """
    bridge.reset_per_call()
    eir_cfg = bridge.eir_cfg
    erd_cfg = bridge.erd_cfg
    if not eir_cfg.enabled or bridge.scorer is None:
        bridge.last_block_reason = EIR_DISABLED
        return {}
    cache = bridge.cache
    if cache is None or getattr(cache, "actions_flat", None) is None:
        bridge.last_block_reason = EIR_NO_CACHE
        return {}
    if bridge.capture is None or not getattr(bridge.capture, "fired", False) or bridge.capture.tensor is None:
        bridge.last_block_reason = EIR_NO_CAPTURE
        return {}

    start, end = _read_minibatch_range(bridge)
    feat = bridge.capture.tensor
    if feat.dim() != 2 or feat.shape[0] != (end - start):
        raise RuntimeError(
            f"[EIR] captured feature shape {tuple(feat.shape)} does not match "
            f"minibatch range ({start},{end})"
        )
    device, dtype = feat.device, feat.dtype
    feat_d = feat.detach()
    actions = cache.actions_flat[start:end].to(device=device, dtype=dtype)
    targets = {k: v[start:end].to(device=device, dtype=dtype) for k, v in cache.targets_flat.items()}
    masks = {k: v[start:end].to(device=device, dtype=dtype) for k, v in cache.masks_flat.items()}

    # ----- scorer training (BCE) -----
    logits = bridge.scorer(feat_d, actions.detach())
    scorer_total = feat.new_zeros(())
    terms = 0
    for key, logit in logits.items():
        if key not in targets:
            continue
        loss, n = _masked_bce(logit, targets[key], masks[key])
        if n > 0:
            scorer_total = scorer_total + loss
            terms += 1
    if terms > 0:
        scorer_total = scorer_total / float(terms)
        bridge.scorer_warmup_iters_done += 1
        bridge.last_scorer_active = 1.0
        bridge.last_scorer_loss = float(scorer_total.detach().item())

    bridge.last_target_valid_fraction = cache.mean_valid_fraction()

    # ----- gating for residual distill -----
    lam = eir_cfg.distill_lambda_for_tol(cache.tol_mean)
    block = EIR_OK
    if lam <= 0.0:
        block = EIR_TOL_ABOVE_START
    elif int(frames) < int(eir_cfg.start_after_frames):
        block = EIR_WARMUP_FRAMES
    elif bridge.scorer_warmup_iters_done < int(eir_cfg.scorer_warmup_iters):
        block = EIR_WARMUP_ITERS
    elif bridge.last_target_valid_fraction < float(eir_cfg.min_target_valid_fraction):
        block = EIR_LOW_VALID_FRACTION

    residual_total = feat.new_zeros(())
    if block == EIR_OK:
        d_hat = cache.d_hat_flat[start:end].to(device=device, dtype=dtype)
        tol = cache.success_tol_flat[start:end].to(device=device, dtype=dtype)
        lifted = (
            cache.lifted_flat[start:end].to(device=device, dtype=dtype)
            if cache.lifted_flat is not None else None
        )
        risky = _build_risky_mask(eir_cfg, d_hat=d_hat, tol=tol, lifted=lifted)
        bridge.last_risky_fraction = float(risky.float().mean().item())
        if bridge.last_risky_fraction < float(eir_cfg.min_risky_fraction) or not risky.any():
            block = EIR_NO_RISKY

    if block == EIR_OK:
        a2c = _get_a2c_network(bridge.agent_ref)
        with torch.no_grad():
            mu_live = a2c.mu(feat_d)
            mu_act = getattr(a2c, "mu_act", None)
            if mu_act is not None:
                mu_live = mu_act(mu_live)
        feat_r = feat_d[risky]
        mu_r = mu_live[risky]
        stage = _stage_for_tol(eir_cfg, cache.tol_mean)
        cand_out = generate_eir_candidates(
            mu_r, cfg=eir_cfg, bank=bridge.bank, archive_bank=bridge.archive_bank,
            feat_r=feat_r, tol_mean=cache.tol_mean, stage=stage,
        )
        # RSSM candidate injection. Preference order:
        #   1. FULL H-step MPPI (rssm_planner + rssm_model + per-rollout belief
        #      cache): samples action sequences, prior-rolls them, scores, and
        #      offers the best sequence's first action as an ``rssm`` candidate.
        #   2. LIGHT single-step reranking (rssm_reranker + predict_fn): rescore
        #      candidate first-actions at a fixed belief.
        # Either way the action is only a CANDIDATE; the EIR reranker still
        # chooses among actor/gaussian/elite/archive/rssm. Self-disables to no
        # pool change when belief/model unavailable, uncertain, or low-margin.
        bridge.last_rssm_candidate_added = 0.0
        bridge.last_mppi_active = 0.0
        bridge.last_mppi_belief_valid = 0.0
        rssm_action = _maybe_rssm_candidate_action(
            bridge, feat_r=feat_r, mu_r=mu_r, cand_out=cand_out,
            start=start, end=end, risky=risky, frames=int(frames),
        )
        if rssm_action is not None:
            ra = rssm_action.to(
                device=cand_out["cand"].device, dtype=cand_out["cand"].dtype
            ).clamp(-1.0, 1.0)[:, None, :]
            Bc = ra.shape[0]
            cand_out["cand"] = torch.cat([cand_out["cand"], ra], dim=1)
            cand_out["source_id"] = torch.cat(
                [cand_out["source_id"],
                 torch.full((Bc, 1), SRC_RSSM, device=ra.device, dtype=torch.long)],
                dim=1,
            )
            zk = torch.zeros(Bc, 1, device=ra.device, dtype=cand_out["priority"].dtype)
            cand_out["priority"] = torch.cat([cand_out["priority"], zk], dim=1)
            cand_out["similarity"] = torch.cat([cand_out["similarity"], zk.clone()], dim=1)
            bridge.last_rssm_candidate_added = 1.0
        rer = rerank_candidates(
            cand=cand_out["cand"], source_id=cand_out["source_id"],
            priority=cand_out["priority"], similarity=cand_out["similarity"],
            scorer=bridge.scorer, feat_r=feat_r, cfg=eir_cfg, erd_cfg=erd_cfg,
        )
        bridge.last_candidate_source_frac = rer["candidate_source_frac"]
        bridge.last_best_source_frac = rer["best_source_frac"]
        bridge.last_elite_similarity_mean = rer["elite_similarity_mean"]
        bridge.last_elite_action_delta_norm = rer["elite_action_delta_norm"]
        bridge.last_score_margin = rer["margin_mean"]
        bridge.last_best_score = rer["best_score_mean"]
        bridge.last_actor_score = rer["actor_score_mean"]

        eligible = rer["eligible"]
        if not bool(eligible.any()):
            block = EIR_NOT_ELIGIBLE
        elif bridge.residual_bridge is not None and bridge.residual_bridge.head is not None:
            feat_e = feat_r[eligible]
            mu_e = mu_r[eligible]
            a_star = rer["best_action"][eligible]
            res_out = compute_recovery_residual_loss(
                bridge.residual_bridge, feat_e=feat_e, mu_live_e=mu_e,
                a_star=a_star, tol_mean=cache.tol_mean,
                best_source_frac_elite=(
                    bridge.last_best_source_frac.get("elite", 0.0)
                    + bridge.last_best_source_frac.get("archive", 0.0)
                ),
                score_margin=bridge.last_score_margin,
            )
            if "eir_residual_total" in res_out:
                residual_total = float(lam) * res_out["eir_residual_total"]
                bridge.last_residual_active = 1.0

    bridge.last_block_reason = int(block)
    if bridge.bank is not None:
        bridge.last_elite_bank_size = float(getattr(bridge.bank, "size", 0))
        bridge.last_bank_metrics = bridge.bank.metrics()
    total = float(eir_cfg.scorer_loss_weight) * scorer_total + residual_total
    bridge.last_total_loss = float(total.detach().item())
    bridge.last_active = 1.0 if (bridge.last_scorer_active > 0 or bridge.last_residual_active > 0) else 0.0
    if total.detach().abs().item() <= 0.0:
        return {}
    return {"eir_total": total}
