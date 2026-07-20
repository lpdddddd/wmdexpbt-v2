"""MSA routing, same-tau retention and the promotion decision.

Statistical contract (pre-registered):
- L_mid = q_mid - q_pre is computed ONLY from per-seed paired results at the
  SAME tau_plus and the SAME ordered seed manifest (fail-closed otherwise).
  Tolerances are heterogeneous across workers: each worker's L_mid uses its
  OWN tau_plus[w] = max(0.010, 0.95 * tau[w]), so L_mid is a
  CRITERION-LOCAL improvement. Cross-worker routing and the panel
  statistic therefore compare criterion-local improvements, not a shared
  absolute criterion (documented on every consumer).
- Second-half routing: workers sorted by L_mid descending, ties broken by
  ascending worker_id (stable/deterministic); top-k receive the second
  packet. The MSA_PERMUTED_ROUTE control arm shuffles the L_mid -> worker
  mapping with a pre-registered PRNG seed BEFORE routing, keeping total
  tau_plus dose, recipient count and evaluation count identical.
- q_post is an unequal-dose follow-up for recipients only; it never enters
  the six-worker common-dose statistic.
- Retention rho = p_post / p_pre at the SAME tau; requires p_pre >= 0.01,
  otherwise retention_unavailable=true and promotion is impossible. The API
  takes a single tau and per-seed results tagged with their tau — comparing
  against any other tolerance's historical peak is structurally impossible.
- Panel statistic: per-worker paired bootstrap over the N paired
  pre/mid tau_plus seeds (resample within worker), then the median of the
  six per-worker L_mid replicates; fixed PRNG seed, 10000 repeats, 5th
  percentile. This is a CONDITIONAL decision statistic for the current
  six-checkpoint panel — it is not a population CI over independent seeds.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

# Pre-registered constants.
BOOTSTRAP_SEED = 20260719
BOOTSTRAP_REPEATS = 10_000
PANEL_QUANTILE = 0.05
PERMUTED_ROUTE_SEED = 777001
MIN_P_PRE_FOR_RETENTION = 0.01
TOP_K_RECIPIENTS = 3


class ManifestMismatch(RuntimeError):
    """pre/mid results differ in tolerance or seed manifest: fail closed."""


@dataclass
class SeedEvalResult:
    """Per-seed fixed-tolerance evaluation outcome for one snapshot."""
    manifest_id: str
    tau: float
    seeds: list[int]                 # ordered seed manifest
    success: list[float]             # per-seed success in [0, 1], same order

    def __post_init__(self) -> None:
        if len(self.seeds) != len(self.success):
            raise ValueError("seeds and success must have equal length")

    @property
    def mean(self) -> float:
        return sum(self.success) / len(self.success) if self.success else float("nan")


def _check_paired(a: SeedEvalResult, b: SeedEvalResult) -> None:
    if a.manifest_id != b.manifest_id:
        raise ManifestMismatch(
            f"seed manifest mismatch: {a.manifest_id} vs {b.manifest_id}")
    if abs(a.tau - b.tau) > 1e-12:
        raise ManifestMismatch(f"tolerance mismatch: {a.tau} vs {b.tau}")
    if a.seeds != b.seeds:
        raise ManifestMismatch("seed order mismatch between paired evals")


def l_mid(pre: SeedEvalResult, mid: SeedEvalResult) -> float:
    """Common-dose learning statistic q_mid - q_pre (same tau_plus, same
    ordered manifest; anything else fails closed)."""
    _check_paired(pre, mid)
    return mid.mean - pre.mean


# ---------------------------------------------------------------------------
# Second-half routing
# ---------------------------------------------------------------------------

@dataclass
class RoutingResult:
    arm: str                            # "MSA" | "MSA_PERMUTED_ROUTE"
    order: list[int]                    # worker ids, L_mid descending
    recipients: list[int]               # top-k receiving the second packet
    l_mid_by_worker: dict[int, float]
    permutation_seed: Optional[int] = None
    permuted_assignment: Optional[dict[int, float]] = None


def route_second_half(
    l_mid_by_worker: dict[int, float],
    top_k: int = TOP_K_RECIPIENTS,
    permuted: bool = False,
    permutation_seed: int = PERMUTED_ROUTE_SEED,
) -> RoutingResult:
    """Sort by (criterion-local) L_mid descending (stable tie-break:
    ascending worker_id) and give the second tau_plus packet (one committed
    20M tick at each recipient's OWN tau_plus) to the top-k workers.

    permuted=True (MSA_PERMUTED_ROUTE): shuffle the mapping between L_mid
    values and worker identities with the pre-registered seed BEFORE
    sorting. Total dose, number of recipients and evaluation count are
    unchanged — only the identity->score mapping is broken.
    """
    if permuted:
        workers = sorted(l_mid_by_worker.keys())
        values = [l_mid_by_worker[w] for w in workers]
        rng = random.Random(permutation_seed)
        shuffled = values[:]
        rng.shuffle(shuffled)
        effective = dict(zip(workers, shuffled))
    else:
        effective = dict(l_mid_by_worker)

    order = sorted(effective.keys(), key=lambda w: (-effective[w], w))
    return RoutingResult(
        arm="MSA_PERMUTED_ROUTE" if permuted else "MSA",
        order=order,
        recipients=order[:top_k],
        l_mid_by_worker=dict(l_mid_by_worker),
        permutation_seed=permutation_seed if permuted else None,
        permuted_assignment=effective if permuted else None,
    )


# ---------------------------------------------------------------------------
# Same-tolerance retention
# ---------------------------------------------------------------------------

@dataclass
class RetentionResult:
    tau: float
    p_pre: float
    p_post: float
    rho: Optional[float]
    retention_unavailable: bool
    reason: str = ""


def same_tau_retention(pre: SeedEvalResult, post: SeedEvalResult) -> RetentionResult:
    """rho = p_post / p_pre at the SAME tau (pre/post snapshots of the same
    worker). p_pre < 0.01 -> retention_unavailable, promotion impossible.

    There is no argument through which a historical peak at another
    tolerance could enter this computation.
    """
    _check_paired(pre, post)
    p_pre, p_post = pre.mean, post.mean
    if p_pre < MIN_P_PRE_FOR_RETENTION:
        return RetentionResult(tau=pre.tau, p_pre=p_pre, p_post=p_post,
                               rho=None, retention_unavailable=True,
                               reason=f"p_pre={p_pre:.4f} < "
                                      f"{MIN_P_PRE_FOR_RETENTION}")
    return RetentionResult(tau=pre.tau, p_pre=p_pre, p_post=p_post,
                           rho=p_post / p_pre, retention_unavailable=False)


# ---------------------------------------------------------------------------
# Paired bootstrap panel statistic
# ---------------------------------------------------------------------------

def panel_bootstrap_lower_quantile(
    paired_diffs_by_worker: dict[int, Sequence[float]],
    n_boot: int = BOOTSTRAP_REPEATS,
    seed: int = BOOTSTRAP_SEED,
    quantile: float = PANEL_QUANTILE,
) -> dict:
    """Per-worker paired bootstrap of the per-seed (mid - pre) diffs, then
    the median across the worker panel; returns the 5th percentile of the
    median replicates.

    CONDITIONAL panel statistic: it quantifies evaluation resampling
    uncertainty for THESE six checkpoints; it is NOT a population CI over
    independent training seeds. Deterministic for a fixed seed.
    """
    workers = sorted(paired_diffs_by_worker.keys())
    if not workers:
        raise ValueError("no workers")
    for w in workers:
        if len(paired_diffs_by_worker[w]) < 2:
            raise ValueError(f"worker {w}: need >=2 paired seed diffs")
    rng = random.Random(seed)
    medians: list[float] = []
    for _ in range(n_boot):
        per_worker_means = []
        for w in workers:
            diffs = paired_diffs_by_worker[w]
            n = len(diffs)
            s = 0.0
            for _ in range(n):
                s += diffs[rng.randrange(n)]
            per_worker_means.append(s / n)
        per_worker_means.sort()
        m = len(per_worker_means)
        mid = m // 2
        medians.append(per_worker_means[mid] if m % 2
                       else 0.5 * (per_worker_means[mid - 1] + per_worker_means[mid]))
    medians.sort()
    idx = min(len(medians) - 1, max(0, int(quantile * len(medians))))
    return {
        "statistic": "panel_median_L_mid",
        "lower_quantile": medians[idx],
        "quantile": quantile,
        "n_boot": n_boot,
        "prng_seed": seed,
        "n_workers": len(workers),
        "note": ("criterion-local conditional panel statistic: each "
                 "worker's L_mid is measured at its OWN tau_plus; the "
                 "panel median conditions on this checkpoint panel and "
                 "is NOT a population CI over independent seeds"),
    }


# ---------------------------------------------------------------------------
# Promotion decision
# ---------------------------------------------------------------------------

@dataclass
class PromotionDecision:
    promote: bool
    first_blocking_gate: str            # "none" when promoted
    panel_lower_quantile: float
    retention: dict                     # worker_id -> RetentionResult-ish dict
    details: dict = field(default_factory=dict)


def decide_promotion(
    panel_stat: dict,
    retention_by_worker: dict[int, RetentionResult],
    retention_threshold: float,
    extra_safety_gates: Optional[dict[str, bool]] = None,
) -> PromotionDecision:
    """Permanent promotion to tau_plus iff, in gate order:

    1. panel lower quantile of L_mid > 0;
    2. every worker's same-tau rho passes the CURRENT applicable original
       base-retention threshold (retention_unavailable blocks);
    3. all remaining safety gates (none of which may involve
       cross-tolerance historical retention) pass.

    Gates are checked in order; the FIRST failing gate is recorded.
    """
    lq = float(panel_stat["lower_quantile"])
    if lq <= 0:
        return PromotionDecision(
            promote=False, first_blocking_gate="panel_lower_quantile",
            panel_lower_quantile=lq,
            retention={w: vars(r) for w, r in retention_by_worker.items()})

    for w in sorted(retention_by_worker.keys()):
        r = retention_by_worker[w]
        if r.retention_unavailable:
            return PromotionDecision(
                promote=False,
                first_blocking_gate=f"retention_unavailable_w{w}",
                panel_lower_quantile=lq,
                retention={x: vars(y) for x, y in retention_by_worker.items()})
        if r.rho is None or r.rho < retention_threshold:
            return PromotionDecision(
                promote=False,
                first_blocking_gate=f"same_tau_retention_w{w}",
                panel_lower_quantile=lq,
                retention={x: vars(y) for x, y in retention_by_worker.items()})

    for gate_name, ok in sorted((extra_safety_gates or {}).items()):
        if "historical" in gate_name or "cross_tolerance" in gate_name:
            raise ValueError(
                f"gate '{gate_name}' suggests a cross-tolerance historical "
                "retention read; MSA forbids these gates by construction")
        if not ok:
            return PromotionDecision(
                promote=False, first_blocking_gate=gate_name,
                panel_lower_quantile=lq,
                retention={x: vars(y) for x, y in retention_by_worker.items()})

    return PromotionDecision(
        promote=True, first_blocking_gate="none",
        panel_lower_quantile=lq,
        retention={x: vars(y) for x, y in retention_by_worker.items()})


# ---------------------------------------------------------------------------
# Worker-local selective promotion (MSA_SELECTIVE_STATEFUL)
# ---------------------------------------------------------------------------

PROMOTION_MODE_GLOBAL = "global_panel"
PROMOTION_MODE_SELECTIVE = "worker_local_selective"
MIN_POPULATION_ANCHORS = 3


def worker_bootstrap_lower_quantile(
    paired_diffs: Sequence[float],
    n_boot: int = BOOTSTRAP_REPEATS,
    seed: int = BOOTSTRAP_SEED,
    quantile: float = PANEL_QUANTILE,
) -> dict:
    """Per-worker paired bootstrap of (mid - pre) seed diffs; returns the
    pre-registered lower quantile of the mean replicates.

    Same PRNG contract as the panel statistic (deterministic for a fixed
    seed). This is a CONDITIONAL CI for THIS worker's paired evals — not a
    population CI over independent training seeds.
    """
    diffs = list(paired_diffs)
    if len(diffs) < 2:
        raise ValueError("need >=2 paired seed diffs for worker bootstrap")
    rng = random.Random(seed)
    means: list[float] = []
    n = len(diffs)
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    idx = min(len(means) - 1, max(0, int(quantile * len(means))))
    return {
        "statistic": "worker_mean_L_mid",
        "point": sum(diffs) / n,
        "lower_quantile": means[idx],
        "quantile": quantile,
        "n_boot": n_boot,
        "prng_seed": seed,
        "n_seeds": n,
        "note": ("criterion-local conditional worker L_mid bootstrap; "
                 "NOT a population CI over independent training seeds"),
    }


@dataclass
class SelectivePromotionDecision:
    """Per-worker promote/stay under worker_local_selective mode."""
    promote_any: bool
    selected_for_promotion: list[int]
    anchor_workers: list[int]
    eligible_by_worker: dict[int, bool]
    rejection_reason_by_worker: dict[int, str]
    l_mid_ci_by_worker: dict[int, dict]
    retention: dict
    panel_lower_quantile: float
    first_blocking_gate: str            # cycle-level diagnostic
    final_tol_by_worker: dict[int, float]
    details: dict = field(default_factory=dict)


def decide_selective_promotion(
    paired_diffs_by_worker: dict[int, Sequence[float]],
    retention_by_worker: dict[int, RetentionResult],
    tau_by_worker: dict[int, float],
    tau_plus_by_worker: dict[int, float],
    retention_threshold: float,
    *,
    population_worker_ids: Sequence[int],
    max_promote: int = TOP_K_RECIPIENTS,
    min_anchors: int = MIN_POPULATION_ANCHORS,
    panel_stat: Optional[dict] = None,
) -> SelectivePromotionDecision:
    """Worker-local selective promotion.

    eligible[w] requires ALL of:
      1. p_pre[w] >= MIN_P_PRE_FOR_RETENTION (via retention_unavailable)
      2. worker L_mid lower confidence bound > 0
      3. rho[w] >= retention_threshold (point gate; no invented rho CI)
      4. eval pairing already enforced by callers (ManifestMismatch)

    Selection: eligible sorted by (-L_mid_lo, -rho, worker_id); take at most
    ``max_promote`` such that at least ``min_anchors`` population workers
    remain unpromoted *this cycle*. ``anchor_workers`` are simply the
    population members not selected in this cycle — not a permanent cohort
    retained across cycles. Panel statistic is recorded as a diagnostic
    only — it is NOT a veto.
    """
    l_mid_ci: dict[int, dict] = {}
    eligible: dict[int, bool] = {}
    reasons: dict[int, str] = {}
    due = sorted(retention_by_worker.keys())
    for w in due:
        r = retention_by_worker[w]
        diffs = list(paired_diffs_by_worker.get(w, []))
        try:
            ci = worker_bootstrap_lower_quantile(diffs)
        except ValueError as exc:
            ci = {"lower_quantile": float("-inf"), "error": str(exc)}
        l_mid_ci[w] = ci
        if r.retention_unavailable or r.p_pre < MIN_P_PRE_FOR_RETENTION:
            eligible[w] = False
            reasons[w] = "p_pre<0.01_or_unavailable"
            continue
        if float(ci.get("lower_quantile", float("-inf"))) <= 0:
            eligible[w] = False
            reasons[w] = "L_mid_lower_quantile<=0"
            continue
        if r.rho is None or r.rho < retention_threshold:
            eligible[w] = False
            reasons[w] = "same_tau_retention"
            continue
        eligible[w] = True
        reasons[w] = ""

    ranked = sorted(
        (w for w in due if eligible[w]),
        key=lambda w: (
            -float(l_mid_ci[w]["lower_quantile"]),
            -(float(retention_by_worker[w].rho or 0.0)),
            w,
        ),
    )
    pop = sorted(int(x) for x in population_worker_ids)
    n_pop = len(pop) if pop else max(len(due), 6)
    # selected <= max_promote AND (n_pop - selected) >= min_anchors
    max_by_anchors = max(0, n_pop - int(min_anchors))
    k = min(int(max_promote), max_by_anchors, len(ranked))
    selected = ranked[:k]
    selected_set = set(selected)
    anchors = [w for w in pop if w not in selected_set]
    if not pop:
        anchors = [w for w in due if w not in selected_set]

    final_tol = {w: float(tau_by_worker[w]) for w in due}
    for w in selected:
        final_tol[w] = float(tau_plus_by_worker[w])

    panel_lq = float((panel_stat or {}).get("lower_quantile", float("nan")))
    if not selected:
        gate = "no_eligible_workers"
    else:
        gate = "none"
    return SelectivePromotionDecision(
        promote_any=bool(selected),
        selected_for_promotion=list(selected),
        anchor_workers=list(anchors),
        eligible_by_worker=dict(eligible),
        rejection_reason_by_worker=dict(reasons),
        l_mid_ci_by_worker=dict(l_mid_ci),
        retention={w: vars(r) for w, r in retention_by_worker.items()},
        panel_lower_quantile=panel_lq,
        first_blocking_gate=gate,
        final_tol_by_worker=final_tol,
        details={
            "promotion_mode": PROMOTION_MODE_SELECTIVE,
            "max_promote": int(max_promote),
            "min_anchors": int(min_anchors),
            "panel_is_diagnostic_only": True,
        },
    )
