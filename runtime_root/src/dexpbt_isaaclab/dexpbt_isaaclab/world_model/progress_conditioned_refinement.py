"""Progress-conditioned refinement for CSER-DexPBT (``cser_dexpbt_progress_refine``).

This module implements the scale-invariant, curriculum-agnostic refinement gate
described in the Progress-Conditioned CSER-DexPBT method.  It is intentionally a
set of *pure* tensor functions plus one dataclass config so that every quantity
can be unit-tested on CPU without a simulator.

Core idea (per PPO rollout row ``t`` on environment ``n``)::

    rho_t   = d_t / (eps_t + eps)                     # normalized error
    v_t     = (d_{t-K} - d_t) / (K * (eps_t + eps))   # normalized recent progress
    rho_{t-i} = d_{t-i} / (eps_{t-i} + eps)           # normalized history
    o_t     = Std(rho_{t-K:t})                        # oscillation

    q_prox  = exp(-rho_t / T_p)
    q_stall = sigmoid((m - v_t) / T_s)
    q_osc   = sigmoid((o_t - o_0) / T_o)
    q_local = 1 - (1 - q_stall)(1 - q_osc)
    r_t     = clip((frames - start_after) / ramp, 0, 1)
    g_t     = r_t * q_prox * q_local * I_history * I_lifted

The gate depends only on the *ratio* ``d / tol`` and on progress measured in
tolerance units, so it is invariant to jointly rescaling ``d_hat`` and
``success_tol``.  There is no dependence on any physical distance constant,
fixed tolerance ladder, or ``tol <= 0.02`` style branch.

The window / oscillation quantities are computed from the ``[T, N]`` rollout
layout in :func:`build_progress_context_tn`, which strictly respects episode
boundaries (``done`` / reset).  Any window that crosses a boundary or spans an
invalid rollout row is marked ``history_valid = False`` and produces a zero
gate, so a previous episode's state can never leak into a new one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ProgressRefinementConfig:
    enabled: bool = False
    history_steps: int = 8
    eps: float = 1e-6

    proximity_temperature: float = 1.0
    progress_margin: float = 0.02
    progress_temperature: float = 0.05

    oscillation_enable: bool = True
    oscillation_center: float = 0.10
    oscillation_temperature: float = 0.05

    start_after_frames: int = 300_000_000
    ramp_frames: int = 300_000_000

    only_lifted: bool = True
    min_gate_for_retrieval: float = 0.05
    max_active_fraction: float = 0.35
    max_active_rows: int = 0

    teacher_margin_temperature: float = 0.05
    min_teacher_confidence: float = 0.05
    detach_gate: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Full range + finiteness validation.  Raises ``ValueError`` on bad config."""

        def _finite(name: str, value: float) -> float:
            v = float(value)
            if v != v or v in (float("inf"), float("-inf")):
                raise ValueError(f"ProgressRefinementConfig.{name} must be finite, got {value!r}")
            return v

        def _pos(name: str, value: float) -> float:
            v = _finite(name, value)
            if v <= 0.0:
                raise ValueError(f"ProgressRefinementConfig.{name} must be > 0, got {value!r}")
            return v

        def _nonneg(name: str, value: float) -> float:
            v = _finite(name, value)
            if v < 0.0:
                raise ValueError(f"ProgressRefinementConfig.{name} must be >= 0, got {value!r}")
            return v

        self.enabled = bool(self.enabled)
        self.history_steps = int(self.history_steps)
        if self.history_steps < 1:
            raise ValueError(
                f"ProgressRefinementConfig.history_steps must be >= 1, got {self.history_steps!r}"
            )
        self.eps = _pos("eps", self.eps)

        self.proximity_temperature = _pos("proximity_temperature", self.proximity_temperature)
        self.progress_margin = _finite("progress_margin", self.progress_margin)
        self.progress_temperature = _pos("progress_temperature", self.progress_temperature)

        self.oscillation_enable = bool(self.oscillation_enable)
        self.oscillation_center = _nonneg("oscillation_center", self.oscillation_center)
        self.oscillation_temperature = _pos("oscillation_temperature", self.oscillation_temperature)

        self.start_after_frames = int(self.start_after_frames)
        if self.start_after_frames < 0:
            raise ValueError(
                f"ProgressRefinementConfig.start_after_frames must be >= 0, got {self.start_after_frames!r}"
            )
        self.ramp_frames = int(self.ramp_frames)
        if self.ramp_frames < 0:
            raise ValueError(
                f"ProgressRefinementConfig.ramp_frames must be >= 0, got {self.ramp_frames!r}"
            )

        self.only_lifted = bool(self.only_lifted)
        self.min_gate_for_retrieval = _finite("min_gate_for_retrieval", self.min_gate_for_retrieval)
        if not (0.0 <= self.min_gate_for_retrieval <= 1.0):
            raise ValueError(
                "ProgressRefinementConfig.min_gate_for_retrieval must be in [0, 1], "
                f"got {self.min_gate_for_retrieval!r}"
            )
        self.max_active_fraction = _finite("max_active_fraction", self.max_active_fraction)
        if not (0.0 <= self.max_active_fraction <= 1.0):
            raise ValueError(
                "ProgressRefinementConfig.max_active_fraction must be in [0, 1], "
                f"got {self.max_active_fraction!r}"
            )
        self.max_active_rows = int(self.max_active_rows)
        if self.max_active_rows < 0:
            raise ValueError(
                f"ProgressRefinementConfig.max_active_rows must be >= 0, got {self.max_active_rows!r}"
            )

        self.teacher_margin_temperature = _pos(
            "teacher_margin_temperature", self.teacher_margin_temperature
        )
        self.min_teacher_confidence = _finite("min_teacher_confidence", self.min_teacher_confidence)
        if not (0.0 <= self.min_teacher_confidence <= 1.0):
            raise ValueError(
                "ProgressRefinementConfig.min_teacher_confidence must be in [0, 1], "
                f"got {self.min_teacher_confidence!r}"
            )
        self.detach_gate = bool(self.detach_gate)


@dataclass
class RefinementGateResult:
    """Per-row gate plus finite, detached diagnostic components.

    All tensors are shape ``[B]`` and lie in ``[0, 1]`` (except ``recent_progress``
    and ``normalized_error`` which are the raw normalized quantities).
    """

    gate: torch.Tensor
    proximity: torch.Tensor
    stall: torch.Tensor
    oscillation_gate: torch.Tensor
    local_difficulty: torch.Tensor
    history_valid: torch.Tensor
    normalized_error: torch.Tensor
    recent_progress: torch.Tensor
    oscillation: torch.Tensor
    time_ramp: float
    nonfinite_count: int


def _sanitize(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Replace non-finite entries with 0 and return (clean, n_nonfinite)."""
    finite = torch.isfinite(x)
    n_bad = int((~finite).sum().item())
    if n_bad:
        x = torch.where(finite, x, torch.zeros_like(x))
    return x, n_bad


def normalized_error(d_hat: torch.Tensor, success_tol: torch.Tensor, eps: float) -> torch.Tensor:
    """rho_t = d_t / (eps_t + eps).  Scale-invariant up to the tiny ``eps`` floor."""
    denom = success_tol.clamp_min(0.0) + float(eps)
    return d_hat / denom.clamp_min(float(eps))


def time_ramp(frames: int, start_after_frames: int, ramp_frames: int) -> float:
    """r_t = clip((frames - start_after) / ramp, 0, 1)."""
    if int(ramp_frames) <= 0:
        return 1.0 if int(frames) >= int(start_after_frames) else 0.0
    r = (float(frames) - float(start_after_frames)) / float(ramp_frames)
    return float(min(1.0, max(0.0, r)))


def proximity_gate(rho: torch.Tensor, temperature: float) -> torch.Tensor:
    """q_prox = exp(-rho / T_p); larger when the state is closer to the boundary."""
    return torch.exp(-rho.clamp_min(0.0) / max(float(temperature), 1e-9))


def stall_gate(recent_progress: torch.Tensor, margin: float, temperature: float) -> torch.Tensor:
    """q_stall = sigmoid((m - v) / T_s); larger when normalized progress is low."""
    return torch.sigmoid((float(margin) - recent_progress) / max(float(temperature), 1e-9))


def oscillation_gate(oscillation: torch.Tensor, center: float, temperature: float) -> torch.Tensor:
    """q_osc = sigmoid((o - o_0) / T_o); larger when the state oscillates more."""
    return torch.sigmoid((oscillation - float(center)) / max(float(temperature), 1e-9))


def local_difficulty(stall: torch.Tensor, oscillation_gate_val: torch.Tensor) -> torch.Tensor:
    """q_local = 1 - (1 - q_stall)(1 - q_osc)."""
    return 1.0 - (1.0 - stall) * (1.0 - oscillation_gate_val)


def build_progress_context_tn(
    d_hat_tn: torch.Tensor,
    success_tol_tn: torch.Tensor,
    dones_tn: torch.Tensor,
    cfg: ProgressRefinementConfig,
    *,
    valid_tn: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Compute per-row progress context from ``[T, N]`` rollout tensors.

    Returns flat tensors using the shared ``flat[n * T + t] = x[t, n]`` layout so
    they align with ``PPODataset.last_range`` and the existing ERD ``*_flat``
    fields.

    Episode boundaries are strict: for a row at time ``t`` the length-``K`` window
    ``[t-K, t-1]`` must contain no ``done`` transition and every step must be a
    valid rollout row, otherwise ``progress_history_valid = 0`` and both the
    ``recent_progress`` and ``oscillation`` derived quantities fall back to
    neutral values (0), which yields a zero gate downstream.

    Keys: ``normalized_error_flat``, ``recent_progress_flat``,
    ``oscillation_flat``, ``progress_history_valid_flat``.
    """
    d = d_hat_tn.float()
    tol = success_tol_tn.float()
    dn = dones_tn.float()
    if d.dim() == 3 and d.shape[-1] == 1:
        d = d.squeeze(-1)
    if tol.dim() == 3 and tol.shape[-1] == 1:
        tol = tol.squeeze(-1)
    if dn.dim() == 3 and dn.shape[-1] == 1:
        dn = dn.squeeze(-1)
    T, N = d.shape
    K = int(cfg.history_steps)
    eps = float(cfg.eps)
    device = d.device

    if valid_tn is None:
        valid = torch.ones(T, N, device=device)
    else:
        valid = valid_tn.float()
        if valid.dim() == 3 and valid.shape[-1] == 1:
            valid = valid.squeeze(-1)

    rho = d / (tol.clamp_min(0.0) + eps).clamp_min(eps)  # [T, N] normalized error

    norm_err = torch.zeros(T, N, device=device)
    recent_prog = torch.zeros(T, N, device=device)
    osc = torch.zeros(T, N, device=device)
    hist_valid = torch.zeros(T, N, device=device)

    for t in range(T):
        norm_err[t] = rho[t]
        if t - K < 0:
            # Not enough history collected in this rollout chunk.
            continue
        # Window of the K steps preceding t: indices [t-K, t-1]; plus current t for
        # the oscillation window [t-K, t].  A boundary is any done in [t-K, t-1]
        # (a done at step j means j and j+1 belong to different episodes) or any
        # invalid rollout row within [t-K, t].
        prev_slice = slice(t - K, t)  # [t-K .. t-1]
        done_in_window = dn[prev_slice].sum(dim=0)  # [N]
        valid_window = valid[t - K : t + 1].prod(dim=0)  # [N] includes current
        ok = (done_in_window <= 0.5) & (valid_window > 0.5)
        hist_valid[t] = ok.float()

        d_tmk = d[t - K]
        v = (d_tmk - d[t]) / (float(K) * (tol[t].clamp_min(0.0) + eps).clamp_min(eps))
        recent_prog[t] = torch.where(ok, v, torch.zeros_like(v))

        window_rho = rho[t - K : t + 1]  # [K+1, N]
        # Population std (unbiased=False) over the window; neutral 0 where invalid.
        o = window_rho.std(dim=0, unbiased=False)
        osc[t] = torch.where(ok, o, torch.zeros_like(o))

    def _flat(x: torch.Tensor) -> torch.Tensor:
        return x.transpose(0, 1).reshape(T * N).contiguous()

    return {
        "normalized_error_flat": _flat(norm_err),
        "recent_progress_flat": _flat(recent_prog),
        "oscillation_flat": _flat(osc),
        "progress_history_valid_flat": _flat(hist_valid),
    }


def compute_refinement_gate(
    *,
    normalized_error: torch.Tensor,
    recent_progress: torch.Tensor,
    oscillation: torch.Tensor,
    history_valid: torch.Tensor,
    frames: int,
    cfg: ProgressRefinementConfig,
    lifted: Optional[torch.Tensor] = None,
) -> RefinementGateResult:
    """Compute the row-wise refinement gate ``g_t`` and its components.

    All inputs are ``[B]`` tensors (already row-aligned to the live minibatch).
    Outputs are finite, shape ``[B]``, and (for the gate/components) in ``[0, 1]``.
    The gate is detached from environment labels when ``cfg.detach_gate`` is set.
    """
    device = normalized_error.device
    B = int(normalized_error.shape[0])
    nonfinite = 0

    rho, nb = _sanitize(normalized_error.float())
    nonfinite += nb
    rho = rho.clamp_min(0.0)
    v, nb = _sanitize(recent_progress.float())
    nonfinite += nb
    o, nb = _sanitize(oscillation.float())
    nonfinite += nb
    o = o.clamp_min(0.0)
    hist = history_valid.float()
    hist = torch.where(torch.isfinite(hist), hist, torch.zeros_like(hist))
    hist = (hist > 0.5).float()

    q_prox = proximity_gate(rho, cfg.proximity_temperature)
    q_stall = stall_gate(v, cfg.progress_margin, cfg.progress_temperature)
    if cfg.oscillation_enable:
        q_osc = oscillation_gate(o, cfg.oscillation_center, cfg.oscillation_temperature)
    else:
        q_osc = torch.zeros(B, device=device)
    q_local = local_difficulty(q_stall, q_osc)

    ramp = time_ramp(int(frames), cfg.start_after_frames, cfg.ramp_frames)

    if cfg.only_lifted:
        if lifted is None:
            lifted_mask = torch.ones(B, device=device)
        else:
            lm = lifted.float()
            lm = torch.where(torch.isfinite(lm), lm, torch.zeros_like(lm))
            lifted_mask = (lm > 0.5).float()
    else:
        lifted_mask = torch.ones(B, device=device)

    gate = float(ramp) * q_prox * q_local * hist * lifted_mask
    gate, nb = _sanitize(gate)
    nonfinite += nb
    gate = gate.clamp(0.0, 1.0)

    if cfg.detach_gate:
        gate = gate.detach()

    return RefinementGateResult(
        gate=gate,
        proximity=q_prox.clamp(0.0, 1.0),
        stall=q_stall.clamp(0.0, 1.0),
        oscillation_gate=q_osc.clamp(0.0, 1.0),
        local_difficulty=q_local.clamp(0.0, 1.0),
        history_valid=hist,
        normalized_error=rho,
        recent_progress=v,
        oscillation=o,
        time_ramp=float(ramp),
        nonfinite_count=int(nonfinite),
    )


def select_retrieval_rows(
    gate: torch.Tensor, cfg: ProgressRefinementConfig
) -> tuple[torch.Tensor, float]:
    """Return a boolean retrieval mask ``[B]`` and the selected budget fraction.

    Only rows with ``gate >= min_gate_for_retrieval`` are eligible; the eligible
    set is then capped by ``max_active_fraction`` (of ``B``) and ``max_active_rows``
    (0 = unlimited), keeping the highest-gate rows.  Budget selection only limits
    which rows perform retrieval; it never mutates the gate statistics.
    """
    B = int(gate.shape[0])
    if B == 0:
        return torch.zeros(0, dtype=torch.bool, device=gate.device), 0.0
    eligible = gate >= float(cfg.min_gate_for_retrieval)
    n_eligible = int(eligible.sum().item())
    if n_eligible == 0:
        return torch.zeros(B, dtype=torch.bool, device=gate.device), 0.0

    budget = n_eligible
    frac_cap = int(float(cfg.max_active_fraction) * B) if cfg.max_active_fraction < 1.0 else B
    if cfg.max_active_fraction < 1.0:
        budget = min(budget, max(1, frac_cap))
    if int(cfg.max_active_rows) > 0:
        budget = min(budget, int(cfg.max_active_rows))

    if budget >= n_eligible:
        mask = eligible
    else:
        masked_gate = torch.where(eligible, gate, torch.full_like(gate, -1.0))
        top_idx = torch.topk(masked_gate, budget, dim=0).indices
        mask = torch.zeros(B, dtype=torch.bool, device=gate.device)
        mask[top_idx] = True
    return mask, float(int(mask.sum().item())) / float(B)


def teacher_confidence(
    *,
    score_best: torch.Tensor,
    score_actor: torch.Tensor,
    cfg: ProgressRefinementConfig,
    wm_uncertainty: Optional[torch.Tensor] = None,
    alpha_wm: float = 0.0,
    retrieval_valid: Optional[torch.Tensor] = None,
    wm_valid: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Continuous teacher confidence ``c_teacher = c_margin * c_wm``.

    ``c_margin = sigmoid((S_best - S_actor) / T_m)`` rewards candidates that beat
    the live actor. ``score_best`` must be the score of the *selected* teacher
    (same ``candidate_score`` / ``best_idx`` used for selection), not an
    independent ``max`` over a different score family.

    ``c_wm = 1 - alpha_wm * clip(uncertainty, 0, 1)`` folds in calibrated
    world-model uncertainty; when the WM is unavailable / not yet calibrated
    (``alpha_wm <= 0``, ``wm_uncertainty is None``, or ``wm_valid`` is false) it
    is exactly 1 (neutral), so invalid WM never looks like uncertainty=0
    high confidence. Rows whose retrieval is invalid receive confidence 0.
    """
    margin = (score_best - score_actor) / max(float(cfg.teacher_margin_temperature), 1e-9)
    c_margin = torch.sigmoid(margin)
    if wm_uncertainty is not None and float(alpha_wm) > 0.0:
        u = wm_uncertainty.float().clamp(0.0, 1.0)
        c_wm = 1.0 - float(alpha_wm) * u
        if wm_valid is not None:
            c_wm = torch.where(
                wm_valid.to(device=c_wm.device, dtype=torch.bool),
                c_wm,
                torch.ones_like(c_wm),
            )
    else:
        c_wm = torch.ones_like(c_margin)
    c = (c_margin * c_wm).clamp(0.0, 1.0)
    if retrieval_valid is not None:
        rv = (retrieval_valid.float() > 0.5).float()
        c = c * rv
    c, _ = _sanitize(c)
    diag = {
        "teacher_margin_mean": float(c_margin.mean().item()) if c.numel() else 0.0,
        "teacher_confidence_mean": float(c.mean().item()) if c.numel() else 0.0,
    }
    return c, diag


def weighted_effective_sample_size(weight: torch.Tensor) -> float:
    """Kish effective sample size: ``(sum w)^2 / sum w^2``."""
    w = weight.float()
    s1 = float(w.sum().item())
    s2 = float((w * w).sum().item())
    if s2 <= 0.0:
        return 0.0
    return (s1 * s1) / s2
