"""RSSM reranking + full H-step MPPI for EIR-MPC.

Two modes share this module:

* LIGHT (``RSSMReranker``): at a FIXED belief, rescore candidate first-actions
  with the trained prediction heads. Single-step, no latent rollout. Kept as a
  fallback and for backward compatibility.

* FULL (``RSSMMPPIPlanner``): a genuine H-step model-predictive action-sequence
  search. Given a per-row belief ``b_t`` (from ``rssm_belief_provider``) it
  samples ``K`` action sequences of length ``H``, rolls them forward through the
  RSSM *prior* transition (``model.imagine_step``), scores each sequence with the
  success/drop/recovery/value/task-error heads, optionally refines with CEM, and
  offers the best sequence's first action back into the EIR candidate pool as an
  ``rssm`` source. The EIR reranker still chooses among actor / gaussian / elite
  / archive / rssm, so MPPI never unilaterally overrides elite/scorer.

Everything runs under ``torch.no_grad`` (no gradient through the world model).
Both modes self-disable with a ``fallback_reason`` when the model / belief is
unavailable, before the warmup frame, on high uncertainty, or on low margin.

If the backbone lacks a real prior transition, ``imagine_step`` degrades to a
fixed-belief rollout and the planner logs ``imagine_mode=fixed_belief_fallback``
so the run is still informative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import torch


# fallback reasons
RERANK_OK = 0
RERANK_DISABLED = 1
RERANK_BEFORE_START = 2
RERANK_MODEL_UNAVAILABLE = 3
RERANK_NO_HEADS = 4
RERANK_HIGH_UNCERTAINTY = 5


@dataclass
class RSSMRerankConfig:
    enabled: bool = False
    start_after_frames: int = 800_000_000
    horizon: int = 4
    num_sequences: int = 64
    noise_std: float = 0.10
    elite_frac: float = 0.2
    uncertainty_threshold: float = 0.4
    min_margin: float = 0.0

    w_success: float = 1.0
    w_recovery: float = 0.8
    w_drop: float = 2.0
    w_value: float = 0.2


def _extract_head_probs(
    preds: dict[str, Any],
    device,
    dtype,
    *,
    horizon_steps: Optional[Sequence[int]] = None,
) -> dict[str, torch.Tensor]:
    """Pull recovery + mean success/drop probs via shared extractor.

    Averaging uses configured ``horizon_steps`` when provided; otherwise all
    numeric horizon keys that expose both success and drop (sorted).
    """
    from .wm_head_probs import extract_head_probs_compat

    return extract_head_probs_compat(
        preds, device, dtype, horizon_steps=horizon_steps
    )


class RSSMReranker:
    def __init__(self, cfg: RSSMRerankConfig) -> None:
        self.cfg = cfg
        self.last_active = 0.0
        self.last_best_score = 0.0
        self.last_actor_score = 0.0
        self.last_score_margin = 0.0
        self.last_uncertainty = 0.0
        self.last_best_first_action_delta = 0.0
        self.last_fallback_reason = RERANK_DISABLED

    def score_candidates(
        self,
        *,
        belief_flat: Optional[torch.Tensor],
        cand_actions: torch.Tensor,
        predict_fn: Optional[Callable[[torch.Tensor, torch.Tensor], dict[str, Any]]],
        value_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        frames: int = 0,
    ) -> dict[str, Any]:
        """Score ``cand_actions[B,K,A]`` at ``belief_flat[B,D]``.

        Returns dict: ``score[B,K]``, ``best_action[B,A]``, ``best_idx[B]``,
        ``uncertainty[B]``, ``active`` (bool), ``fallback_reason`` (int).
        Candidate column 0 is treated as the actor action for margin.
        """
        cfg = self.cfg
        B, K, A = cand_actions.shape
        dev = cand_actions.device
        dtype = cand_actions.dtype

        def _inactive(reason: int) -> dict[str, Any]:
            self.last_active = 0.0
            self.last_fallback_reason = int(reason)
            return {
                "score": torch.zeros(B, K, device=dev, dtype=dtype),
                "best_action": cand_actions[:, 0, :],
                "best_idx": torch.zeros(B, dtype=torch.long, device=dev),
                "uncertainty": torch.zeros(B, device=dev, dtype=dtype),
                "active": False,
                "fallback_reason": int(reason),
            }

        if not cfg.enabled:
            return _inactive(RERANK_DISABLED)
        if int(frames) < int(cfg.start_after_frames):
            return _inactive(RERANK_BEFORE_START)
        if predict_fn is None or belief_flat is None:
            return _inactive(RERANK_MODEL_UNAVAILABLE)

        with torch.no_grad():
            belief_rep = belief_flat[:, None, :].expand(B, K, belief_flat.shape[-1]).reshape(B * K, -1)
            cand_flat = cand_actions.reshape(B * K, A)
            try:
                preds = predict_fn(belief_rep, cand_flat)
            except Exception:
                return _inactive(RERANK_MODEL_UNAVAILABLE)
            probs = _extract_head_probs(preds, dev, dtype)
            if "success" not in probs and "recovery" not in probs:
                return _inactive(RERANK_NO_HEADS)

            succ = probs.get("success", torch.zeros(B * K, device=dev, dtype=dtype)).view(B, K)
            drop = probs.get("drop", torch.zeros(B * K, device=dev, dtype=dtype)).view(B, K)
            rec = probs.get("recovery", torch.zeros(B * K, device=dev, dtype=dtype)).view(B, K)

            val_term = torch.zeros(B, K, device=dev, dtype=dtype)
            if value_fn is not None:
                try:
                    v = value_fn(belief_rep).reshape(B, K)
                    v_min = v.min(dim=1, keepdim=True).values
                    v_max = v.max(dim=1, keepdim=True).values
                    val_term = (v - v_min) / (v_max - v_min + 1e-6)
                except Exception:
                    val_term = torch.zeros(B, K, device=dev, dtype=dtype)

            score = (
                cfg.w_success * succ
                + cfg.w_recovery * rec
                - cfg.w_drop * drop
                + cfg.w_value * val_term
            )

            # uncertainty proxy: std of success prob across horizons (per row,
            # averaged over candidates). High => unreliable, fall back.
            if "success_stack" in probs and int(probs["success_stack"].shape[0]) >= 2:
                ss = probs["success_stack"].view(-1, B, K)  # [H, B, K]
                uncertainty = ss.std(dim=0).mean(dim=1)  # [B]
            else:
                # single horizon (or none): no cross-horizon disagreement signal.
                uncertainty = torch.zeros(B, device=dev, dtype=dtype)

            best_idx = score.argmax(dim=1)
            rows = torch.arange(B, device=dev)
            best_action = cand_actions[rows, best_idx]
            best_score = score[rows, best_idx]
            actor_score = score[:, 0]
            margin = best_score - actor_score
            first_delta = (best_action - cand_actions[:, 0, :]).norm(dim=-1)

            self.last_best_score = float(best_score.mean().item())
            self.last_actor_score = float(actor_score.mean().item())
            self.last_score_margin = float(margin.mean().item())
            self.last_uncertainty = float(uncertainty.mean().item())
            self.last_best_first_action_delta = float(first_delta.mean().item())

            if float(uncertainty.mean().item()) > float(cfg.uncertainty_threshold):
                self.last_active = 0.0
                self.last_fallback_reason = RERANK_HIGH_UNCERTAINTY
                return {
                    "score": score,
                    "best_action": cand_actions[:, 0, :],
                    "best_idx": torch.zeros(B, dtype=torch.long, device=dev),
                    "uncertainty": uncertainty,
                    "active": False,
                    "fallback_reason": RERANK_HIGH_UNCERTAINTY,
                }

            self.last_active = 1.0
            self.last_fallback_reason = RERANK_OK
            return {
                "score": score,
                "best_action": best_action,
                "best_idx": best_idx,
                "uncertainty": uncertainty,
                "active": True,
                "fallback_reason": RERANK_OK,
            }


# ======================================================================
# Full H-step MPPI planner
# ======================================================================

# MPPI fallback reasons.
MPPI_OK = 0
MPPI_DISABLED = 1
MPPI_BEFORE_START = 2
MPPI_NO_BELIEF = 3
MPPI_NO_RSSM_MODEL = 4
MPPI_NO_IMAGINE_STEP = 5
MPPI_HIGH_UNCERTAINTY = 6
MPPI_LOW_MARGIN = 7
MPPI_EXCEPTION = 8

_MPPI_FALLBACK_NAMES = {
    MPPI_OK: "ok", MPPI_DISABLED: "disabled", MPPI_BEFORE_START: "before_start",
    MPPI_NO_BELIEF: "no_belief", MPPI_NO_RSSM_MODEL: "no_rssm_model",
    MPPI_NO_IMAGINE_STEP: "no_imagine_step", MPPI_HIGH_UNCERTAINTY: "high_uncertainty",
    MPPI_LOW_MARGIN: "low_margin", MPPI_EXCEPTION: "exception",
}

# Ensemble MPPI fallback reasons (for eir_mppi/ensemble_fallback_reason).
ENSEMBLE_FB_NOT_CONFIGURED = 0
ENSEMBLE_FB_NO_CACHE = 1
ENSEMBLE_FB_BELOW_MIN_VALID = 2
ENSEMBLE_FB_SCORING_USED = 3
ENSEMBLE_FB_PLANNER_INACTIVE = 4
ENSEMBLE_FB_ACTIVE = 5

_ENSEMBLE_FB_NAMES = {
    ENSEMBLE_FB_NOT_CONFIGURED: "not_configured",
    ENSEMBLE_FB_NO_CACHE: "no_cache",
    ENSEMBLE_FB_BELOW_MIN_VALID: "below_min_valid",
    ENSEMBLE_FB_SCORING_USED: "scoring_used",
    ENSEMBLE_FB_PLANNER_INACTIVE: "planner_inactive",
    ENSEMBLE_FB_ACTIVE: "active",
}


@dataclass
class RSSMMPPIConfig:
    enabled: bool = False
    mode: str = "full"  # off | light | full
    start_after_frames: int = 800_000_000

    horizon: int = 4
    num_sequences: int = 64
    num_elite: int = 8
    cem_iters: int = 1

    action_noise_std: float = 0.10
    action_noise_decay: float = 0.85
    max_action_delta: float = 0.5
    smooth_noise: bool = True

    include_actor_sequence: bool = True
    # ``include_elite_sequence`` / ``include_archive_sequence``: when True, repeat
    # non-actor EIR candidate first-actions as fixed H-step sequences. The planner
    # does not receive per-candidate ``source_id`` yet, so elite vs archive filtering
    # is not implemented; all candidate columns except index 0 (actor) are included.
    include_elite_sequence: bool = True
    include_archive_sequence: bool = True  # reserved; same path as elite until source_id wired

    uncertainty_threshold: float = 0.4
    min_sequence_margin: float = 0.02

    w_success: float = 1.0
    w_recovery: float = 0.8
    w_drop: float = 2.0
    w_value: float = 0.2
    w_task_improve: float = 0.5
    w_uncertainty: float = 0.2
    w_action_smoothness: float = 0.05
    w_action_delta: float = 0.02

    fail_fast: bool = False

    # Stochastic risk-sensitive MPPI (default off -> deterministic path unchanged).
    stochastic_rollouts: bool = False
    num_latent_samples: int = 1
    sample_temperature: float = 1.0
    risk_beta: float = 0.0
    drop_cvar_alpha: float = 0.25
    drop_cvar_weight: float = 0.0
    uncertainty_cvar_weight: float = 0.0
    score_aggregation: str = "mean"  # mean | mean_minus_std | cvar
    max_rollout_batch: int = 65536

    # Full RSSM ensemble for MPPI scoring only (default off -> single-model
    # behaviour unchanged). Disagreement across independent RSSM members adds an
    # epistemic-uncertainty penalty so a single over-optimistic model cannot
    # push a harmful SRC_RSSM candidate.
    use_model_ensemble: bool = False
    ensemble_size: int = 1
    ensemble_risk_beta: float = 0.0
    ensemble_drop_weight: float = 0.0
    ensemble_uncertainty_weight: float = 0.0
    ensemble_min_valid_models: int = 2
    ensemble_subsample_models: bool = False
    ensemble_max_models_per_plan: int = 3
    # Ensemble worst-case / CVaR uncertainty lives on a different scale than the
    # single-model ``prior_std_mean`` (it is a per-member max over latent CVaR),
    # so it cannot share the single-model absolute gate ``uncertainty_threshold``.
    # The ensemble path uses ``ensemble_uncertainty_threshold`` instead. When the
    # uncertainty is below this threshold but the candidate beats the actor by at
    # least ``ensemble_uncertainty_allow_margin``, the gate lets the SRC_RSSM
    # candidate through even if uncertainty is elevated.
    ensemble_uncertainty_threshold: float = 2.0
    ensemble_uncertainty_allow_margin: float = 0.10

    def __post_init__(self) -> None:
        self.horizon = max(1, int(self.horizon))
        self.num_sequences = max(1, int(self.num_sequences))
        self.num_elite = max(1, min(int(self.num_elite), int(self.num_sequences)))
        self.cem_iters = max(1, int(self.cem_iters))
        self.num_latent_samples = max(1, int(self.num_latent_samples))
        self.max_rollout_batch = max(1, int(self.max_rollout_batch))
        self.ensemble_size = max(1, int(self.ensemble_size))
        self.ensemble_min_valid_models = max(1, int(self.ensemble_min_valid_models))
        self.ensemble_max_models_per_plan = max(1, int(self.ensemble_max_models_per_plan))
        agg = str(self.score_aggregation or "mean").strip().lower()
        if agg not in ("mean", "mean_minus_std", "cvar"):
            agg = "mean"
        self.score_aggregation = agg


def _tail_mean(x: torch.Tensor, alpha: float, largest: bool) -> torch.Tensor:
    """Tail mean over the last dimension of ``x`` shaped ``[..., M]``.

    When ``largest=True`` returns the mean of the top ``ceil(alpha*M)`` values
    (worst-case tail for drop/uncertainty). When ``largest=False`` returns the
    mean of the bottom tail (conservative score CVaR).
    """
    if x.dim() < 2:
        raise ValueError(f"_tail_mean expects at least 2 dims; got {tuple(x.shape)}")
    m = int(x.shape[-1])
    k = max(1, int(math.ceil(float(alpha) * m)))
    vals, _ = torch.topk(x, k, dim=-1, largest=bool(largest))
    return vals.mean(dim=-1)


class RSSMMPPIPlanner:
    """Full H-step model-predictive action-sequence search over RSSM prior."""

    def __init__(self, cfg: RSSMMPPIConfig) -> None:
        self.cfg = cfg
        self.last_active = 0.0
        self.last_fallback_reason = MPPI_DISABLED
        self.last_mode = cfg.mode
        self.last_imagine_mode = "none"
        self.last_best_score = 0.0
        self.last_actor_score = 0.0
        self.last_score_margin = 0.0
        self.last_raw_score_margin = 0.0
        self.last_uncertainty = 0.0
        self.last_best_first_action_delta = 0.0
        self.last_low_margin_fraction = 0.0
        self.last_high_uncertainty_fraction = 0.0
        self.last_num_sequences = 0
        self.last_horizon = 0
        self.last_stochastic_rollouts = 0.0
        self.last_num_latent_samples = 0.0
        self.last_sample_temperature = 0.0
        self.last_score_std = 0.0
        self.last_drop_cvar = 0.0
        self.last_uncertainty_cvar = 0.0
        self.last_risk_adjusted_margin = 0.0
        self.last_risk_beta = 0.0
        self.last_drop_cvar_weight = 0.0
        # ensemble MPPI scoring telemetry
        self.last_use_model_ensemble = 0.0
        self.last_ensemble_size = 0.0
        self.last_ensemble_valid_size = 0.0
        self.last_ensemble_score_std = 0.0
        self.last_ensemble_drop_worst = 0.0
        self.last_ensemble_uncertainty_worst = 0.0
        self.last_ensemble_risk_beta = 0.0
        self.last_ensemble_drop_weight = 0.0
        self.last_ensemble_uncertainty_weight = 0.0
        self.last_ensemble_configured = 0.0
        self.last_ensemble_available_size = 0.0
        self.last_ensemble_used = 0.0
        self.last_ensemble_fallback_reason = float(ENSEMBLE_FB_NOT_CONFIGURED)
        # ensemble-specific uncertainty gate telemetry
        self.last_effective_uncertainty_threshold = 0.0
        self.last_ensemble_uncertainty_threshold = 0.0
        self.last_ensemble_uncertainty_allow_margin = 0.0
        self.last_uncertainty_gate_allow_by_margin = 0.0
        self.last_uncertainty_gate_blocked = 0.0

    # ---- helpers --------------------------------------------------------
    def _inactive(self, reason: int, B: int, A: int, actor_action: torch.Tensor, dev, dtype) -> dict:
        self.last_active = 0.0
        self.last_fallback_reason = int(reason)
        H = int(self.cfg.horizon)
        return {
            "active": False,
            "fallback_reason": int(reason),
            "best_action": actor_action,
            "best_sequence": actor_action[:, None, :].expand(B, H, A).contiguous(),
            "best_score": torch.zeros(B, device=dev, dtype=dtype),
            "actor_score": torch.zeros(B, device=dev, dtype=dtype),
            "score_margin": torch.zeros(B, device=dev, dtype=dtype),
            "uncertainty": torch.zeros(B, device=dev, dtype=dtype),
            "best_first_action_delta": torch.zeros(B, device=dev, dtype=dtype),
            "mode": self.cfg.mode,
            "imagine_mode": "none",
        }

    def _mark_ensemble_planner_outcome(self, *, active: bool) -> None:
        if self.last_ensemble_used >= 1.0:
            self.last_ensemble_fallback_reason = float(
                ENSEMBLE_FB_ACTIVE if active else ENSEMBLE_FB_PLANNER_INACTIVE
            )

    @staticmethod
    def _evaluate_uncertainty_gate(
        *,
        use_ensemble: bool,
        mean_unc: float,
        mean_margin: float,
        cfg: "RSSMMPPIConfig",
    ) -> dict:
        """Pure uncertainty-gate decision shared by ``plan()`` and tests.

        The single-model path keeps the original absolute ``uncertainty_threshold``
        (default 0.4) so legacy behaviour is unchanged. The ensemble path uses
        ``ensemble_uncertainty_threshold`` instead, because the ensemble
        worst-case / CVaR uncertainty is on a different (larger) scale than the
        single-model ``prior_std_mean``. As a margin override, when an ensemble
        candidate beats the actor by at least ``ensemble_uncertainty_allow_margin``
        and stays at/under the ensemble threshold, it is explicitly allowed.

        Returns a dict with ``effective_threshold``, ``ensemble_threshold``,
        ``ensemble_allow_margin``, ``allow_by_margin`` and ``blocked``.
        """
        if use_ensemble:
            effective_threshold = float(cfg.ensemble_uncertainty_threshold)
        else:
            effective_threshold = float(cfg.uncertainty_threshold)
        allow_by_margin = bool(
            use_ensemble
            and mean_margin >= float(cfg.ensemble_uncertainty_allow_margin)
            and mean_unc <= float(cfg.ensemble_uncertainty_threshold)
        )
        blocked = bool(mean_unc > effective_threshold and not allow_by_margin)
        return {
            "effective_threshold": effective_threshold,
            "ensemble_threshold": float(cfg.ensemble_uncertainty_threshold),
            "ensemble_allow_margin": float(cfg.ensemble_uncertainty_allow_margin),
            "allow_by_margin": allow_by_margin,
            "blocked": blocked,
        }

    def _build_sequences(
        self,
        actor_action: torch.Tensor,        # [B, A]
        candidate_actions: torch.Tensor,   # [B, K, A]
        mean_seq: torch.Tensor,            # [B, H, A] gaussian mean
        std: float,
    ) -> tuple[torch.Tensor, int]:
        """Return ``(seq[B, Ktot, H, A], actor_index)``. Column 0 is the actor."""
        cfg = self.cfg
        B, A = actor_action.shape
        H = int(cfg.horizon)
        dev, dtype = actor_action.device, actor_action.dtype
        blocks: list[torch.Tensor] = []

        # actor repeated sequence (always index 0)
        actor_seq = actor_action[:, None, :].expand(B, H, A).unsqueeze(1)  # [B,1,H,A]
        blocks.append(actor_seq)

        # candidate-derived repeated sequences (skip column 0 = actor baseline)
        if cfg.include_elite_sequence and candidate_actions is not None and candidate_actions.shape[1] > 1:
            cand_for_seq = candidate_actions[:, 1:, :]
            cand_seq = cand_for_seq[:, :, None, :].expand(
                B, cand_for_seq.shape[1], H, A
            )
            blocks.append(cand_seq)

        # sampled gaussian sequences around the (CEM) mean
        K = int(cfg.num_sequences)
        if K > 0:
            noise = torch.randn(B, K, H, A, device=dev, dtype=dtype) * float(std)
            if cfg.smooth_noise and H > 1:
                noise = torch.cumsum(noise, dim=2) / (H ** 0.5)
            noise = noise.clamp(-float(cfg.max_action_delta), float(cfg.max_action_delta))
            sampled = (mean_seq[:, None, :, :] + noise).clamp(-1.0, 1.0)
            blocks.append(sampled)

        seq = torch.cat(blocks, dim=1).clamp(-1.0, 1.0)  # [B, Ktot, H, A]
        return seq, 0

    def _flat_rollout_scores(
        self,
        *,
        belief_flat: torch.Tensor,
        seq_flat: torch.Tensor,
        actor_action_flat: torch.Tensor,
        rssm_model,
        deterministic: bool,
        sample_temperature: float,
    ) -> dict[str, torch.Tensor | str]:
        """Roll ``seq_flat[N,H,A]`` from ``belief_flat[N,D]``; return per-row scores."""
        cfg = self.cfg
        n, h, a = int(seq_flat.shape[0]), int(seq_flat.shape[1]), int(seq_flat.shape[2])
        dev, dtype = seq_flat.device, seq_flat.dtype
        state = rssm_model.get_state_from_belief(belief_flat)
        succ_steps: list[torch.Tensor] = []
        drop_steps: list[torch.Tensor] = []
        rec_steps: list[torch.Tensor] = []
        value_steps: list[torch.Tensor] = []
        task_delta_steps: list[torch.Tensor] = []
        unc_steps: list[torch.Tensor] = []
        imagine_mode = "none"

        for step in range(h):
            action_h = seq_flat[:, step, :]
            state, preds = rssm_model.imagine_step(
                state,
                action_h,
                deterministic=deterministic,
                sample_temperature=float(sample_temperature),
            )
            imagine_mode = str(preds.get("imagine_mode", imagine_mode))
            zeros = torch.zeros(n, device=dev, dtype=dtype)
            succ_steps.append(preds.get("success", zeros).reshape(n).to(dtype))
            drop_steps.append(preds.get("drop", zeros).reshape(n).to(dtype))
            rec_steps.append(preds.get("recovery", zeros).reshape(n).to(dtype))
            value_steps.append(preds.get("value", zeros).reshape(n).to(dtype))
            task_delta_steps.append(preds.get("task_error_delta", zeros).reshape(n).to(dtype))
            unc_steps.append(preds.get("prior_std_mean", zeros).reshape(n).to(dtype))

        succ = torch.stack(succ_steps, dim=0)
        drop = torch.stack(drop_steps, dim=0)
        rec = torch.stack(rec_steps, dim=0)
        value = torch.stack(value_steps, dim=0)
        task_delta = torch.stack(task_delta_steps, dim=0)
        unc = torch.stack(unc_steps, dim=0)

        succ_max = succ.max(dim=0).values
        drop_max = drop.max(dim=0).values
        rec_mean = rec.mean(dim=0)
        value_last = value[-1]
        task_improve = -task_delta.sum(dim=0)
        unc_mean = unc.mean(dim=0)

        actor_rep = actor_action_flat
        a_delta = ((seq_flat - actor_rep[:, None, :]) ** 2).sum(dim=-1).sum(dim=1)
        if h > 1:
            smooth = ((seq_flat[:, 1:, :] - seq_flat[:, :-1, :]) ** 2).sum(dim=-1).sum(dim=1)
        else:
            smooth = torch.zeros(n, device=dev, dtype=dtype)

        score = (
            cfg.w_success * succ_max
            + cfg.w_recovery * rec_mean
            - cfg.w_drop * drop_max
            + cfg.w_value * value_last
            + cfg.w_task_improve * task_improve
            - cfg.w_uncertainty * unc_mean
            - cfg.w_action_smoothness * smooth
            - cfg.w_action_delta * a_delta
        )
        return {
            "score": score,
            "drop_max": drop_max,
            "unc_mean": unc_mean,
            "imagine_mode": imagine_mode,
        }

    def _rollout_batch(
        self,
        *,
        belief_flat: torch.Tensor,
        seq_flat: torch.Tensor,
        actor_action_flat: torch.Tensor,
        rssm_model,
        deterministic: bool,
        sample_temperature: float,
    ) -> dict[str, torch.Tensor | str]:
        """Chunked wrapper around ``_flat_rollout_scores`` for large ``N``."""
        cfg = self.cfg
        n = int(seq_flat.shape[0])
        chunk = int(cfg.max_rollout_batch)
        if n <= chunk:
            return self._flat_rollout_scores(
                belief_flat=belief_flat,
                seq_flat=seq_flat,
                actor_action_flat=actor_action_flat,
                rssm_model=rssm_model,
                deterministic=deterministic,
                sample_temperature=sample_temperature,
            )
        scores: list[torch.Tensor] = []
        drops: list[torch.Tensor] = []
        uncs: list[torch.Tensor] = []
        imagine_mode = "none"
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            part = self._flat_rollout_scores(
                belief_flat=belief_flat[start:end],
                seq_flat=seq_flat[start:end],
                actor_action_flat=actor_action_flat[start:end],
                rssm_model=rssm_model,
                deterministic=deterministic,
                sample_temperature=sample_temperature,
            )
            scores.append(part["score"])
            drops.append(part["drop_max"])
            uncs.append(part["unc_mean"])
            imagine_mode = str(part.get("imagine_mode", imagine_mode))
        return {
            "score": torch.cat(scores, dim=0),
            "drop_max": torch.cat(drops, dim=0),
            "unc_mean": torch.cat(uncs, dim=0),
            "imagine_mode": imagine_mode,
        }

    def _aggregate_sample_scores(
        self,
        *,
        score_samples: torch.Tensor,
        drop_samples: torch.Tensor,
        unc_samples: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Risk-sensitive aggregation over latent samples ``[..., M]``."""
        cfg = self.cfg
        b, k, m = score_samples.shape
        score_mean = score_samples.mean(dim=-1)
        if m > 1:
            score_std = score_samples.std(dim=-1, unbiased=False)
            drop_cvar = _tail_mean(drop_samples, cfg.drop_cvar_alpha, largest=True)
            unc_cvar = _tail_mean(unc_samples, cfg.drop_cvar_alpha, largest=True)
        else:
            score_std = torch.zeros(b, k, device=score_samples.device, dtype=score_samples.dtype)
            drop_cvar = drop_samples.squeeze(-1)
            unc_cvar = unc_samples.squeeze(-1)

        if cfg.score_aggregation == "cvar":
            score_final = _tail_mean(score_samples, cfg.drop_cvar_alpha, largest=False)
        elif cfg.score_aggregation == "mean_minus_std":
            score_final = (
                score_mean
                - float(cfg.risk_beta) * score_std
                - float(cfg.drop_cvar_weight) * drop_cvar
                - float(cfg.uncertainty_cvar_weight) * unc_cvar
            )
        else:
            score_final = score_mean

        unc_final = unc_cvar if m > 1 else unc_samples.squeeze(-1)
        return {
            "score": score_final,
            # raw mean over latent samples, BEFORE single-model risk penalties.
            # Used by the ensemble path to avoid double-counting risk terms.
            "score_mean": score_mean,
            "uncertainty": unc_final,
            "score_std": score_std,
            "drop_cvar": drop_cvar,
            "uncertainty_cvar": unc_cvar,
        }

    @torch.no_grad()
    def _rollout_and_score(
        self, *, belief_flat: torch.Tensor, seq: torch.Tensor, actor_action: torch.Tensor, rssm_model
    ) -> dict:
        """Roll each [B,Ktot,H,A] sequence through the prior and score it."""
        cfg = self.cfg
        b, k, h, a = seq.shape
        dev, dtype = seq.device, seq.dtype
        d = int(belief_flat.shape[-1])
        use_stochastic = bool(cfg.stochastic_rollouts and int(cfg.num_latent_samples) > 1)
        m = int(cfg.num_latent_samples) if use_stochastic else 1
        deterministic = not use_stochastic

        if use_stochastic:
            belief_rep = (
                belief_flat[:, None, None, :]
                .expand(b, k, m, d)
                .reshape(b * k * m, d)
            )
            seq_flat = (
                seq[:, :, None, :, :]
                .expand(b, k, m, h, a)
                .reshape(b * k * m, h, a)
            )
            actor_rep = (
                actor_action[:, None, None, :]
                .expand(b, k, m, a)
                .reshape(b * k * m, a)
            )
        else:
            belief_rep = belief_flat[:, None, :].expand(b, k, d).reshape(b * k, d)
            seq_flat = seq.reshape(b * k, h, a)
            actor_rep = actor_action[:, None, :].expand(b, k, a).reshape(b * k, a)

        flat = self._rollout_batch(
            belief_flat=belief_rep,
            seq_flat=seq_flat,
            actor_action_flat=actor_rep,
            rssm_model=rssm_model,
            deterministic=deterministic,
            sample_temperature=float(cfg.sample_temperature),
        )
        imagine_mode = str(flat.get("imagine_mode", "none"))

        if use_stochastic:
            score_s = flat["score"].reshape(b, k, m)
            drop_s = flat["drop_max"].reshape(b, k, m)
            unc_s = flat["unc_mean"].reshape(b, k, m)
            agg = self._aggregate_sample_scores(
                score_samples=score_s, drop_samples=drop_s, unc_samples=unc_s
            )
            return {
                "score": agg["score"],
                "score_mean": agg["score_mean"],
                "uncertainty": agg["uncertainty"],
                "score_std": agg["score_std"],
                "drop_cvar": agg["drop_cvar"],
                "uncertainty_cvar": agg["uncertainty_cvar"],
                "imagine_mode": imagine_mode,
            }

        score = flat["score"].reshape(b, k)
        unc_bk = flat["unc_mean"].reshape(b, k)
        zeros_bk = torch.zeros(b, k, device=dev, dtype=dtype)
        return {
            "score": score,
            "score_mean": score,
            "uncertainty": unc_bk,
            "score_std": zeros_bk,
            "drop_cvar": flat["drop_max"].reshape(b, k),
            "uncertainty_cvar": unc_bk,
            "imagine_mode": imagine_mode,
        }

    @torch.no_grad()
    def _rollout_and_score_ensemble(
        self,
        *,
        belief_flat_list: list[torch.Tensor],
        seq: torch.Tensor,
        actor_action: torch.Tensor,
        rssm_models: list,
    ) -> dict:
        """Score each sequence with an RSSM ensemble (model-disagreement).

        Each member's per-sequence ``score_mean`` (raw mean over latent
        samples, no single-model risk penalty) and its latent std / drop /
        uncertainty are collected, then risk terms are applied ONCE across the
        ensemble so there is no double-counting with the single-model path.
        """
        cfg = self.cfg
        b, k, _h, _a = seq.shape
        dev, dtype = seq.device, seq.dtype

        score_list: list[torch.Tensor] = []
        latent_std_list: list[torch.Tensor] = []
        drop_list: list[torch.Tensor] = []
        unc_list: list[torch.Tensor] = []
        imagine_mode = "none"
        for model_e, belief_e in zip(rssm_models, belief_flat_list):
            out_e = self._rollout_and_score(
                belief_flat=belief_e,
                seq=seq,
                actor_action=actor_action,
                rssm_model=model_e,
            )
            score_list.append(out_e["score_mean"])  # raw mean, no risk penalty
            latent_std_list.append(out_e["score_std"])
            drop_list.append(out_e["drop_cvar"])
            unc_list.append(out_e["uncertainty"])
            imagine_mode = str(out_e.get("imagine_mode", imagine_mode))

        e = len(score_list)
        score_stack = torch.stack(score_list, dim=-1)          # [B,K,E]
        latent_std_stack = torch.stack(latent_std_list, dim=-1)
        drop_stack = torch.stack(drop_list, dim=-1)
        unc_stack = torch.stack(unc_list, dim=-1)

        score_mean = score_stack.mean(dim=-1)                  # [B,K]
        ensemble_score_std = (
            score_stack.std(dim=-1, unbiased=False)
            if e > 1 else torch.zeros(b, k, device=dev, dtype=dtype)
        )
        latent_score_std = latent_std_stack.mean(dim=-1)
        drop_ensemble_worst = drop_stack.max(dim=-1).values
        unc_ensemble_worst = unc_stack.max(dim=-1).values

        score_final = (
            score_mean
            - float(cfg.risk_beta) * latent_score_std
            - float(cfg.ensemble_risk_beta) * ensemble_score_std
            - float(cfg.drop_cvar_weight) * drop_ensemble_worst
            - float(cfg.ensemble_drop_weight) * drop_ensemble_worst
            - float(cfg.uncertainty_cvar_weight) * unc_ensemble_worst
            - float(cfg.ensemble_uncertainty_weight) * unc_ensemble_worst
        )

        stochastic = "stochastic" in imagine_mode
        ens_mode = (
            "ensemble_stochastic_prior_rollout" if stochastic
            else "ensemble_prior_rollout"
        )
        return {
            "score": score_final,
            "uncertainty": unc_ensemble_worst,
            "score_std": latent_score_std,
            "ensemble_score_std": ensemble_score_std,
            "drop_cvar": drop_ensemble_worst,
            "uncertainty_cvar": unc_ensemble_worst,
            "ensemble_valid_size": e,
            "imagine_mode": ens_mode,
        }

    # ---- main entry -----------------------------------------------------
    @torch.no_grad()
    def plan(
        self,
        *,
        belief_flat: Optional[torch.Tensor],
        actor_action: torch.Tensor,
        candidate_actions: torch.Tensor,
        rssm_model,
        frames: int,
        belief_flat_ensemble: Optional[list] = None,
        rssm_models: Optional[list] = None,
    ) -> dict:
        cfg = self.cfg
        B, A = actor_action.shape
        dev, dtype = actor_action.device, actor_action.dtype
        self.last_mode = cfg.mode
        self.last_num_sequences = 0
        self.last_horizon = int(cfg.horizon)
        self.last_stochastic_rollouts = 1.0 if (
            cfg.stochastic_rollouts and int(cfg.num_latent_samples) > 1
        ) else 0.0
        self.last_num_latent_samples = float(cfg.num_latent_samples)
        self.last_sample_temperature = float(cfg.sample_temperature)
        self.last_risk_beta = float(cfg.risk_beta)
        self.last_drop_cvar_weight = float(cfg.drop_cvar_weight)
        self.last_use_model_ensemble = 0.0
        self.last_ensemble_size = float(cfg.ensemble_size)
        self.last_ensemble_valid_size = 0.0
        self.last_ensemble_risk_beta = float(cfg.ensemble_risk_beta)
        self.last_ensemble_drop_weight = float(cfg.ensemble_drop_weight)
        self.last_ensemble_uncertainty_weight = float(cfg.ensemble_uncertainty_weight)
        self.last_ensemble_configured = 1.0 if cfg.use_model_ensemble else 0.0
        self.last_ensemble_available_size = 0.0
        self.last_ensemble_used = 0.0
        self.last_ensemble_fallback_reason = float(
            ENSEMBLE_FB_NOT_CONFIGURED if not cfg.use_model_ensemble else ENSEMBLE_FB_NO_CACHE
        )
        self.last_effective_uncertainty_threshold = float(cfg.uncertainty_threshold)
        self.last_ensemble_uncertainty_threshold = float(cfg.ensemble_uncertainty_threshold)
        self.last_ensemble_uncertainty_allow_margin = float(cfg.ensemble_uncertainty_allow_margin)
        self.last_uncertainty_gate_allow_by_margin = 0.0
        self.last_uncertainty_gate_blocked = 0.0

        if not cfg.enabled or cfg.mode == "off":
            return self._inactive(MPPI_DISABLED, B, A, actor_action, dev, dtype)
        if int(frames) < int(cfg.start_after_frames):
            return self._inactive(MPPI_BEFORE_START, B, A, actor_action, dev, dtype)
        if belief_flat is None:
            return self._inactive(MPPI_NO_BELIEF, B, A, actor_action, dev, dtype)
        if rssm_model is None or not hasattr(rssm_model, "get_state_from_belief"):
            return self._inactive(MPPI_NO_RSSM_MODEL, B, A, actor_action, dev, dtype)
        if not hasattr(rssm_model, "imagine_step"):
            return self._inactive(MPPI_NO_IMAGINE_STEP, B, A, actor_action, dev, dtype)
        if int(belief_flat.shape[0]) != B:
            return self._inactive(MPPI_NO_BELIEF, B, A, actor_action, dev, dtype)

        # ---- decide single-model vs ensemble scoring ----
        ens_models: list = []
        ens_beliefs: list = []
        use_ensemble = False
        if cfg.use_model_ensemble:
            if rssm_models is None or belief_flat_ensemble is None:
                self.last_ensemble_fallback_reason = float(ENSEMBLE_FB_NO_CACHE)
            elif (
                len(rssm_models) != len(belief_flat_ensemble)
                or len(rssm_models) <= 1
            ):
                self.last_ensemble_fallback_reason = float(ENSEMBLE_FB_NO_CACHE)
            else:
                for model_e, belief_e in zip(rssm_models, belief_flat_ensemble):
                    if (
                        model_e is not None
                        and belief_e is not None
                        and hasattr(model_e, "get_state_from_belief")
                        and hasattr(model_e, "imagine_step")
                        and int(belief_e.shape[0]) == B
                    ):
                        ens_models.append(model_e)
                        ens_beliefs.append(belief_e.to(device=dev, dtype=dtype))
                self.last_ensemble_available_size = float(len(ens_models))
                if cfg.ensemble_subsample_models and len(ens_models) > int(cfg.ensemble_max_models_per_plan):
                    ens_models = ens_models[: int(cfg.ensemble_max_models_per_plan)]
                    ens_beliefs = ens_beliefs[: int(cfg.ensemble_max_models_per_plan)]
                use_ensemble = len(ens_models) >= int(cfg.ensemble_min_valid_models)
                if use_ensemble:
                    self.last_ensemble_used = 1.0
                    self.last_ensemble_fallback_reason = float(ENSEMBLE_FB_SCORING_USED)
                elif len(ens_models) > 0:
                    self.last_ensemble_fallback_reason = float(ENSEMBLE_FB_BELOW_MIN_VALID)
                else:
                    self.last_ensemble_fallback_reason = float(ENSEMBLE_FB_NO_CACHE)

        def _score_seq(seq_in: torch.Tensor) -> dict:
            if use_ensemble:
                return self._rollout_and_score_ensemble(
                    belief_flat_list=ens_beliefs,
                    seq=seq_in,
                    actor_action=actor_action,
                    rssm_models=ens_models,
                )
            return self._rollout_and_score(
                belief_flat=belief_flat, seq=seq_in, actor_action=actor_action,
                rssm_model=rssm_model,
            )

        if use_ensemble:
            self.last_use_model_ensemble = 1.0
            self.last_ensemble_valid_size = float(len(ens_models))

        try:
            belief_flat = belief_flat.to(device=dev, dtype=dtype)
            H = int(cfg.horizon)
            mean_seq = actor_action[:, None, :].expand(B, H, A).contiguous()
            std = float(cfg.action_noise_std)
            last = None
            seq = None
            for it in range(int(cfg.cem_iters)):
                seq, _actor_idx = self._build_sequences(
                    actor_action, candidate_actions, mean_seq, std
                )
                last = _score_seq(seq)
                if it < int(cfg.cem_iters) - 1:
                    # CEM refit: take top num_elite, update mean over their steps,
                    # decay std, then resample on the next iter.
                    score = last["score"]
                    K = int(score.shape[1])
                    ne = min(int(cfg.num_elite), K)
                    elite_idx = torch.topk(score, ne, dim=1).indices  # [B, ne]
                    rows = torch.arange(B, device=dev)[:, None].expand(B, ne)
                    elite_seq = seq[rows, elite_idx]  # [B, ne, H, A]
                    mean_seq = elite_seq.mean(dim=1)  # [B, H, A]
                    std = std * float(cfg.action_noise_decay)

            score = last["score"]              # [B, Ktot]
            unc = last["uncertainty"]          # [B, Ktot]
            score_std_bk = last.get("score_std")
            drop_cvar_bk = last.get("drop_cvar")
            unc_cvar_bk = last.get("uncertainty_cvar")
            ensemble_score_std_bk = last.get("ensemble_score_std")
            self.last_imagine_mode = last["imagine_mode"]
            self.last_num_sequences = int(seq.shape[1])

            actor_score = score[:, 0]
            best_idx = score.argmax(dim=1)
            rows = torch.arange(B, device=dev)
            best_score = score[rows, best_idx]
            best_seq = seq[rows, best_idx]            # [B, H, A]
            best_action = best_seq[:, 0, :]
            margin = best_score - actor_score
            # ``score`` is already risk-adjusted in ``_aggregate_sample_scores`` when
            # stochastic / mean_minus_std / CVaR penalties are enabled.
            raw_score_margin = margin
            risk_adjusted_margin = margin
            best_unc = unc[rows, best_idx]
            first_delta = (best_action - actor_action).norm(dim=-1)

            self.last_best_score = float(best_score.mean().item())
            self.last_actor_score = float(actor_score.mean().item())
            self.last_score_margin = float(margin.mean().item())
            self.last_raw_score_margin = float(raw_score_margin.mean().item())
            self.last_uncertainty = float(best_unc.mean().item())
            self.last_best_first_action_delta = float(first_delta.mean().item())
            if isinstance(score_std_bk, torch.Tensor):
                self.last_score_std = float(score_std_bk[rows, best_idx].mean().item())
            else:
                self.last_score_std = 0.0
            if isinstance(drop_cvar_bk, torch.Tensor):
                self.last_drop_cvar = float(drop_cvar_bk[rows, best_idx].mean().item())
            else:
                self.last_drop_cvar = 0.0
            if isinstance(unc_cvar_bk, torch.Tensor):
                self.last_uncertainty_cvar = float(unc_cvar_bk[rows, best_idx].mean().item())
            else:
                self.last_uncertainty_cvar = 0.0
            self.last_risk_adjusted_margin = float(risk_adjusted_margin.mean().item())
            if use_ensemble and isinstance(ensemble_score_std_bk, torch.Tensor):
                self.last_ensemble_score_std = float(ensemble_score_std_bk[rows, best_idx].mean().item())
                self.last_ensemble_drop_worst = float(
                    drop_cvar_bk[rows, best_idx].mean().item()
                    if isinstance(drop_cvar_bk, torch.Tensor) else 0.0
                )
                self.last_ensemble_uncertainty_worst = float(best_unc.mean().item())
            self.last_high_uncertainty_fraction = float(
                (best_unc > float(cfg.uncertainty_threshold)).float().mean().item()
            )
            self.last_low_margin_fraction = float(
                (margin < float(cfg.min_sequence_margin)).float().mean().item()
            )

            mean_unc = float(best_unc.mean().item())
            mean_margin = float(margin.mean().item())
            # uncertainty gate (only meaningful with a real prior rollout; in
            # fixed-belief fallback prior_std is 0 so this never blocks). The
            # single-model path keeps the absolute ``uncertainty_threshold``; the
            # ensemble path uses ``ensemble_uncertainty_threshold`` (different
            # scale) with an optional high-margin override. The decision is made
            # by ``_evaluate_uncertainty_gate`` so plan() and tests stay in sync.
            gate = self._evaluate_uncertainty_gate(
                use_ensemble=use_ensemble,
                mean_unc=mean_unc,
                mean_margin=mean_margin,
                cfg=cfg,
            )
            self.last_effective_uncertainty_threshold = float(gate["effective_threshold"])
            self.last_uncertainty_gate_allow_by_margin = 1.0 if gate["allow_by_margin"] else 0.0
            if gate["blocked"]:
                self.last_uncertainty_gate_blocked = 1.0
                self.last_active = 0.0
                self.last_fallback_reason = MPPI_HIGH_UNCERTAINTY
                out = self._inactive(MPPI_HIGH_UNCERTAINTY, B, A, actor_action, dev, dtype)
                out.update({
                    "uncertainty": best_unc, "score_margin": margin,
                    "risk_adjusted_score_margin": risk_adjusted_margin,
                    "score_std": score_std_bk[rows, best_idx] if isinstance(score_std_bk, torch.Tensor) else None,
                    "drop_cvar": drop_cvar_bk[rows, best_idx] if isinstance(drop_cvar_bk, torch.Tensor) else None,
                    "best_score": best_score, "actor_score": actor_score,
                    "imagine_mode": self.last_imagine_mode,
                })
                self._mark_ensemble_planner_outcome(active=False)
                return out
            if mean_margin < float(cfg.min_sequence_margin):
                self.last_active = 0.0
                self.last_fallback_reason = MPPI_LOW_MARGIN
                out = self._inactive(MPPI_LOW_MARGIN, B, A, actor_action, dev, dtype)
                out.update({
                    "uncertainty": best_unc, "score_margin": margin,
                    "risk_adjusted_score_margin": risk_adjusted_margin,
                    "score_std": score_std_bk[rows, best_idx] if isinstance(score_std_bk, torch.Tensor) else None,
                    "drop_cvar": drop_cvar_bk[rows, best_idx] if isinstance(drop_cvar_bk, torch.Tensor) else None,
                    "best_score": best_score, "actor_score": actor_score,
                    "best_action": best_action, "best_sequence": best_seq,
                    "best_first_action_delta": first_delta,
                    "imagine_mode": self.last_imagine_mode,
                })
                self._mark_ensemble_planner_outcome(active=False)
                return out

            self.last_active = 1.0
            self.last_fallback_reason = MPPI_OK
            self._mark_ensemble_planner_outcome(active=True)
            return {
                "active": True,
                "fallback_reason": MPPI_OK,
                "best_action": best_action,
                "best_sequence": best_seq,
                "best_score": best_score,
                "actor_score": actor_score,
                "score_margin": margin,
                "risk_adjusted_score_margin": risk_adjusted_margin,
                "score_std": score_std_bk[rows, best_idx] if isinstance(score_std_bk, torch.Tensor) else None,
                "drop_cvar": drop_cvar_bk[rows, best_idx] if isinstance(drop_cvar_bk, torch.Tensor) else None,
                "uncertainty_cvar": unc_cvar_bk[rows, best_idx] if isinstance(unc_cvar_bk, torch.Tensor) else None,
                "uncertainty": best_unc,
                "best_first_action_delta": first_delta,
                "mode": cfg.mode,
                "imagine_mode": self.last_imagine_mode,
                "use_model_ensemble": float(self.last_use_model_ensemble),
                "ensemble_valid_size": float(self.last_ensemble_valid_size),
                "ensemble_score_std": (
                    ensemble_score_std_bk[rows, best_idx]
                    if isinstance(ensemble_score_std_bk, torch.Tensor) else None
                ),
            }
        except Exception as exc:  # pragma: no cover - defensive
            if cfg.fail_fast:
                raise
            print(f"[EIR-MPPI] plan exception (fallback): {exc}")
            return self._inactive(MPPI_EXCEPTION, B, A, actor_action, dev, dtype)
