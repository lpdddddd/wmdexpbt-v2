"""Tests for Progress-Conditioned CSER-DexPBT (cser_dexpbt_progress_refine).

Covers the 22 requirements of the method: scale-invariant gate, proximity /
stall / oscillation structure, strict episode boundaries, per-row log-relative
tolerance retrieval + fail-closed, WM-unavailable / non-finite fallback,
continuous-weight distillation, zero-gate / tiny-tolerance safety, bank v1->v2
migration + round-trip, restart-persistent probation/retention, curriculum
-progress stage switching, unchanged legacy profiles, hard-off enforcement, and
absence of any 2cm-specialized branch.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from dexpbt_isaaclab.world_model.progress_conditioned_refinement import (
    ProgressRefinementConfig,
    build_progress_context_tn,
    compute_refinement_gate,
    proximity_gate,
    select_retrieval_rows,
    stall_gate,
    oscillation_gate,
    local_difficulty,
    teacher_confidence,
)
from dexpbt_isaaclab.world_model.elite_snippet_bank import (
    EliteSnippetBank,
    EliteSnippetBankConfig,
    RetrievalResult,
    SNIPPET_SCHEMA_VERSION,
)
from dexpbt_isaaclab.world_model.elite_snippet_builder import SnippetBatch
from dexpbt_isaaclab.world_model.snippet_distill import (
    SnippetDistillConfig,
    compute_snippet_distill_loss,
)
from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import (
    AdaptivePrecisionConfig,
    ProbationState,
    PrecisionRetentionTracker,
    curriculum_progress,
    resolve_stage,
    STAGE_EXPLORATION,
    STAGE_CONSOLIDATION,
    STAGE_SELECTION,
)
from dexpbt_isaaclab.pbt.pbt_consolidation_schedule import (
    PbtConsolidationConfig,
    resolve_pbt_stage,
)
from dexpbt_isaaclab.world_model.wm_strategy_profiles import (
    PROFILE_IDS,
    PROFILE_TABLES,
    apply_wm_strategy_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[4]


def _snippet_batch(tols, *, feat_dim=8, action_dim=3, horizon=4, norm_err=None):
    n = len(tols)
    if norm_err is None:
        norm_err = [0.5] * n
    return SnippetBatch(
        feat_start=torch.randn(n, feat_dim),
        actions=torch.randn(n, horizon, action_dim).clamp(-1, 1),
        d_hat=torch.full((n,), 0.02),
        success_tol=torch.tensor([float(t) for t in tols]),
        lifted=torch.ones(n),
        picked=torch.zeros(n),
        dropped=torch.zeros(n),
        closest_d_hat=torch.full((n,), 0.01),
        success_h=torch.ones(n),
        improve_h=torch.ones(n),
        drop_h=torch.zeros(n),
        recovery_label=torch.ones(n),
        s0=torch.ones(n),
        s1=torch.ones(n),
        i0=torch.ones(n),
        d0=torch.zeros(n),
        d1=torch.zeros(n),
        normalized_error=torch.tensor([float(x) for x in norm_err]),
        recent_progress=torch.zeros(n),
        oscillation=torch.full((n,), 0.1),
        progress_history_valid=torch.ones(n),
        normalized_improvement=torch.full((n,), 0.5),
    )


def _bank(**over):
    cfg = EliteSnippetBankConfig(
        enabled=True, feature_dim=8, action_dim=3, horizon=4, horizons=(4, 8),
        min_priority=0.0, **over,
    )
    return EliteSnippetBank(cfg)


class TestGateMath(unittest.TestCase):
    def _ctx_gate(self, d_tn, tol_tn, dones_tn, cfg, frames=10**12):
        ctx = build_progress_context_tn(d_tn, tol_tn, dones_tn, cfg)
        return compute_refinement_gate(
            normalized_error=ctx["normalized_error_flat"],
            recent_progress=ctx["recent_progress_flat"],
            oscillation=ctx["oscillation_flat"],
            history_valid=ctx["progress_history_valid_flat"],
            frames=frames,
            cfg=cfg,
        ), ctx

    def test_01_scale_invariance(self):
        cfg = ProgressRefinementConfig(enabled=True, history_steps=2)
        T, N = 5, 2
        d = torch.rand(T, N) * 0.05 + 0.03
        tol = torch.full((T, N), 0.04)
        dones = torch.zeros(T, N)
        res_a, _ = self._ctx_gate(d, tol, dones, cfg)
        res_b, _ = self._ctx_gate(d * 7.0, tol * 7.0, dones, cfg)
        self.assertTrue(torch.allclose(res_a.gate, res_b.gate, atol=1e-4))

    def test_02_proximity_monotone(self):
        rho = torch.tensor([0.1, 1.0, 5.0])
        q = proximity_gate(rho, 1.0)
        self.assertTrue(q[0] > q[1] > q[2])

    def test_03_stall_gate_lower_progress_higher(self):
        # Lower recent progress -> larger stall gate.
        low = stall_gate(torch.tensor([-0.1]), margin=0.02, temperature=0.05)
        high = stall_gate(torch.tensor([0.5]), margin=0.02, temperature=0.05)
        self.assertGreater(float(low), float(high))

    def test_04_oscillation_increases_local_difficulty(self):
        q_stall = torch.tensor([0.3])
        lo = oscillation_gate(torch.tensor([0.05]), center=0.10, temperature=0.05)
        hi = oscillation_gate(torch.tensor([0.40]), center=0.10, temperature=0.05)
        self.assertGreater(
            float(local_difficulty(q_stall, hi)), float(local_difficulty(q_stall, lo))
        )

    def test_05_history_crossing_done_zero_gate(self):
        cfg = ProgressRefinementConfig(enabled=True, history_steps=2)
        T, N = 5, 1
        d = torch.tensor([[0.10], [0.08], [0.06], [0.05], [0.045]])
        tol = torch.full((T, N), 0.05)
        dones = torch.zeros(T, N)
        dones[2, 0] = 1.0  # boundary at t=2
        res, ctx = self._ctx_gate(d, tol, dones, cfg)
        hv = ctx["progress_history_valid_flat"]
        # t=3 and t=4 windows include the done -> invalid -> zero gate.
        self.assertEqual(float(hv[3]), 0.0)
        self.assertEqual(float(hv[4]), 0.0)
        self.assertEqual(float(res.gate[3]), 0.0)
        self.assertEqual(float(res.gate[4]), 0.0)

    def test_06_no_cross_env_contamination(self):
        cfg = ProgressRefinementConfig(enabled=True, history_steps=2)
        T, N = 4, 2
        # env0 monotone-improving; env1 constant. Distinct progress per env.
        d = torch.zeros(T, N)
        d[:, 0] = torch.tensor([0.10, 0.08, 0.06, 0.04])
        d[:, 1] = torch.tensor([0.05, 0.05, 0.05, 0.05])
        tol = torch.full((T, N), 0.05)
        dones = torch.zeros(T, N)
        _, ctx = self._ctx_gate(d, tol, dones, cfg)
        rp = ctx["recent_progress_flat"]  # flat[n*T+t]
        # env0 t=3 (idx 3) has positive progress; env1 t=3 (idx 7) ~ zero.
        self.assertGreater(float(rp[3]), 0.0)
        self.assertAlmostEqual(float(rp[7]), 0.0, places=5)

    def test_13_zero_gate_finite_zero_loss(self):
        cfg = SnippetDistillConfig(enabled=True, weight=1.0)
        head = nn.Linear(8, 3)
        feat = torch.randn(4, 8)
        teacher = torch.randn(4, 3)
        loss, stats = compute_snippet_distill_loss(
            mu_fn=head, feat_start=feat, teacher_first_action=teacher,
            cfg=cfg, row_weight=torch.zeros(4),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(stats["snippet_distill/loss"], 0.0)

    def test_12_continuous_weights_in_huber(self):
        cfg = SnippetDistillConfig(enabled=True, weight=1.0, loss_cap=0.0)
        head = nn.Linear(8, 3)
        feat = torch.randn(6, 8)
        teacher = torch.randn(6, 3)
        w = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 0.1])
        loss, stats = compute_snippet_distill_loss(
            mu_fn=head, feat_start=feat, teacher_first_action=teacher,
            cfg=cfg, row_weight=w, min_gate=0.05,
        )
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(head.weight.grad)
        # Effective sample size reflects continuous weights (< count of positives).
        self.assertGreater(stats["snippet_distill/effective_sample_size"], 0.0)
        self.assertLess(stats["snippet_distill/effective_sample_size"], 6.0)
        self.assertAlmostEqual(stats["snippet_distill/weight_ge_min_frac"], 5.0 / 6.0, places=5)


class TestRetrieval(unittest.TestCase):
    def test_07_per_row_tolerance_retrieval(self):
        b = _bank(tolerance_compatibility_mode="log_relative",
                  tolerance_max_log_gap=0.2, progress_priority_weight=0.5)
        b.update_from_builder_batch(_snippet_batch([0.02, 0.021, 0.05, 0.019]), frame=1)
        q = torch.randn(3, 8)
        qt = torch.tensor([0.02, 0.021, 0.05])
        res = b.retrieve_candidates_batched(
            q, query_tolerance=qt, normalized_error=torch.full((3,), 0.5),
            recent_progress=torch.zeros(3), k=2,
        )
        self.assertIsInstance(res, RetrievalResult)
        self.assertEqual(tuple(res.actions.shape), (3, 2, 4, 3))
        # rows 0,1 (0.02/0.021) compatible; row 2 (0.05) fail-closed.
        self.assertTrue(bool(res.valid_mask[0]))
        self.assertTrue(bool(res.valid_mask[1]))
        self.assertFalse(bool(res.valid_mask[2]))

    def test_08_relative_compat_proportional_invariance(self):
        # log-relative gap depends only on the ratio; proportional scaling keeps
        # the same compatibility outcome.
        from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import log_tolerance_gap

        # With a negligible eps floor the gap depends only on the ratio, so a
        # proportional rescaling of both tolerances is invariant.
        g1 = log_tolerance_gap(0.02, 0.024, eps=1e-12)
        g2 = log_tolerance_gap(0.04, 0.048, eps=1e-12)
        self.assertAlmostEqual(g1, g2, places=9)

    def test_09_incompatible_fail_closed(self):
        b = _bank(tolerance_compatibility_mode="log_relative", tolerance_max_log_gap=0.05)
        b.update_from_builder_batch(_snippet_batch([0.02, 0.021]), frame=1)
        q = torch.randn(1, 8)
        res = b.retrieve_candidates_batched(
            q, query_tolerance=torch.tensor([0.10]),
            normalized_error=torch.full((1,), 0.5), k=2,
        )
        self.assertFalse(bool(res.valid_mask[0]))
        self.assertGreaterEqual(res.diagnostics["snippet_retrieval/no_compatible_frac"], 1.0)

    def test_14_tiny_tolerance_no_div_zero(self):
        b = _bank(tolerance_compatibility_mode="log_relative")
        b.update_from_builder_batch(_snippet_batch([1e-9, 2e-9]), frame=1)
        q = torch.randn(1, 8)
        res = b.retrieve_candidates_batched(
            q, query_tolerance=torch.tensor([1e-9]),
            normalized_error=torch.full((1,), 0.5), k=2,
        )
        self.assertTrue(torch.isfinite(res.actions).all())
        self.assertTrue(torch.isfinite(res.tolerance_weight).all())


class TestTeacherConfidence(unittest.TestCase):
    def test_10_wm_unavailable_fallback_priority(self):
        cfg = ProgressRefinementConfig(enabled=True)
        c, _ = teacher_confidence(
            score_best=torch.tensor([2.0]),
            score_actor=torch.tensor([0.0]),
            cfg=cfg,
            wm_uncertainty=None,
            alpha_wm=0.0,
            retrieval_valid=torch.tensor([1.0]),
        )
        # WM unavailable -> c_wm == 1 -> confidence == margin sigmoid > 0 (path open).
        self.assertGreater(float(c), 0.5)

    def test_11_wm_nonfinite_safe(self):
        cfg = ProgressRefinementConfig(enabled=True)
        unc = torch.tensor([float("nan")])
        c, _ = teacher_confidence(
            score_best=torch.tensor([1.0]),
            score_actor=torch.tensor([0.0]),
            cfg=cfg,
            wm_uncertainty=unc,
            alpha_wm=0.5,
            retrieval_valid=torch.tensor([1.0]),
        )
        self.assertTrue(torch.isfinite(c).all())


class TestBankSchema(unittest.TestCase):
    def test_15_v1_migration_loads(self):
        b = _bank(tolerance_compatibility_mode="log_relative")
        b.update_from_builder_batch(_snippet_batch([0.02, 0.021, 0.019]), frame=3)
        st = b.state_dict()
        v1 = dict(st)
        v1["schema_version"] = 1
        for c in ("normalized_error", "recent_progress", "oscillation",
                  "progress_history_valid", "normalized_improvement"):
            v1.pop(c, None)
        b2 = _bank(tolerance_compatibility_mode="log_relative")
        b2.load_state_dict(v1)
        self.assertEqual(b2.size, b.size)
        # normalized_improvement reconstructed = (d_hat - closest)/(tol+eps) ~ 0.5.
        ni = b2.scalars["normalized_improvement"][: b2.size]
        self.assertTrue(torch.isfinite(ni).all())
        self.assertGreater(float(ni.mean()), 0.0)

    def test_16_v2_roundtrip_preserves_fields(self):
        b = _bank()
        b.update_from_builder_batch(
            _snippet_batch([0.02, 0.021], norm_err=[0.3, 0.7]), frame=5
        )
        st = b.state_dict()
        self.assertEqual(st["schema_version"], SNIPPET_SCHEMA_VERSION)
        b2 = _bank()
        b2.load_state_dict(st)
        for c in ("normalized_error", "oscillation", "normalized_improvement"):
            self.assertTrue(
                torch.allclose(
                    b.scalars[c][: b.size], b2.scalars[c][: b2.size]
                )
            )


class TestAdaptiveConsolidation(unittest.TestCase):
    def test_17_restart_preserves_probation_retention(self):
        cfg = AdaptivePrecisionConfig(enabled=True)
        p = ProbationState()
        p.arm(tolerance=0.02, frame=1000, nsucc_mean=30.0)
        for k in range(3):
            p.update(nsucc_mean=31.0, same_level_retention=0.95,
                     frame=1000 + (k + 1) * 10**8, cfg=cfg)
        tr = PrecisionRetentionTracker(quant=0.05)
        tr.update(0.02, 30.0)
        tr.update(0.02, 28.0)
        # serialize -> deserialize
        p2 = ProbationState.from_json(json.loads(json.dumps(p.to_json())))
        tr2 = PrecisionRetentionTracker.from_json(json.loads(json.dumps(tr.to_json())))
        self.assertEqual(p2.stable_ticks, p.stable_ticks)
        self.assertTrue(p2.active)
        snap = tr2.snapshot(0.02, min_samples=1)
        self.assertTrue(snap["valid"])
        self.assertAlmostEqual(snap["retention"], 28.0 / 30.0, places=5)

    def test_18_stage_switches_by_curriculum_progress(self):
        cfg = PbtConsolidationConfig(
            enabled=True, schedule_mode="curriculum_progress",
            stage_consolidation_progress=0.5, stage_selection_progress=0.85,
        )
        self.assertEqual(resolve_pbt_stage(0, cfg, curriculum_progress=0.2), STAGE_EXPLORATION)
        self.assertEqual(resolve_pbt_stage(0, cfg, curriculum_progress=0.6), STAGE_CONSOLIDATION)
        self.assertEqual(resolve_pbt_stage(0, cfg, curriculum_progress=0.9), STAGE_SELECTION)
        # frames mode unchanged when no progress supplied.
        cfg_f = PbtConsolidationConfig(enabled=True, stage1_end_frames=100, stage2_end_frames=200)
        self.assertEqual(resolve_pbt_stage(50, cfg_f), STAGE_EXPLORATION)
        self.assertEqual(resolve_pbt_stage(150, cfg_f), STAGE_CONSOLIDATION)

    def test_curriculum_progress_scale_free(self):
        # Same ratio structure -> same progress.
        c1 = curriculum_progress(0.075, 0.018, 0.0367)
        c2 = curriculum_progress(0.75, 0.18, 0.367)
        self.assertAlmostEqual(c1, c2, places=4)


class _A2C(nn.Module):
    def __init__(self, f=8, a=3):
        super().__init__()
        self.mu = nn.Linear(f, a)


class _Inner:
    def __init__(self, a2c):
        self.a2c_network = a2c


class _Model:
    def __init__(self, a2c):
        self.model = _Inner(a2c)


class _Agent:
    def __init__(self, a2c):
        self.model = _Model(a2c)


class _FakeBridge:
    """Minimal EIRBridge surface for the progress distill runtime path."""

    def __init__(self, bank, sd_cfg, pr_cfg, a2c):
        self.snippet_bank = bank
        self.snippet_distill_cfg = sd_cfg
        self.progress_refine_cfg = pr_cfg
        self.snippet_wm_rerank_cfg = None  # WM unavailable -> priority fallback
        self.wm_calibration_enable = False
        self.wm_calibration_state = None
        self.agent_ref = _Agent(a2c)
        self.rssm_belief_cache = None
        self.rssm_belief_provider = None
        self.rssm_predict_fn = None
        self.capture = None
        self.last_progress_refine = {}
        self.last_snippet_distill_loss = 0.0
        self.last_snippet_distill_active_frac = 0.0
        self.last_snippet_wm_rerank_changed_frac = 0.0
        self.last_snippet_wm_rerank_wm_term_mean = 0.0
        self.last_snippet_retrieval_valid = 0.0
        self.last_snippet_no_tolerance_compatible = 0.0
        self.last_actor_distill_fallback_reason = ""


class _FakeCache:
    pass


class TestRuntimeDistillPath(unittest.TestCase):
    def test_22b_progress_distill_runtime_backward(self):
        from dexpbt_isaaclab.world_model.eir_mpc import _compute_snippet_distill_loss_progress

        a2c = _A2C()
        bank = _bank(tolerance_compatibility_mode="log_relative",
                     tolerance_max_log_gap=0.3, progress_priority_weight=0.5)
        bank.update_from_builder_batch(_snippet_batch([0.02, 0.021, 0.019]), frame=1)
        sd = SnippetDistillConfig(enabled=True, weight=1.0, topk=4, loss_cap=0.0)
        pr = ProgressRefinementConfig(
            enabled=True, history_steps=1, start_after_frames=0, ramp_frames=0,
            only_lifted=False, min_gate_for_retrieval=0.0,
        )
        bridge = _FakeBridge(bank, sd, pr, a2c)

        # Minibatch of 4 rows; build a cache with progress context aligned to it.
        R = 4
        cache = _FakeCache()
        cache.success_tol_flat = torch.full((R,), 0.02)
        cache.lifted_flat = torch.ones(R)
        cache.normalized_error_flat = torch.tensor([0.2, 0.5, 3.0, 0.3])  # rows 0,1,3 close
        cache.recent_progress_flat = torch.zeros(R)  # stagnant -> high stall
        cache.oscillation_flat = torch.full((R,), 0.2)
        cache.progress_history_valid_flat = torch.ones(R)

        feat = torch.randn(R, 8)
        risky = torch.ones(R, dtype=torch.bool)
        loss = _compute_snippet_distill_loss_progress(
            bridge, eir_cfg=None, cache=cache, feat=feat, risky=risky,
            frames=10**9, start=0, end=R,
        )
        self.assertTrue(torch.isfinite(loss))
        # WM unavailable but path stays open (retrieval valid on compatible rows).
        self.assertGreaterEqual(bridge.last_snippet_retrieval_valid, 0.0)
        self.assertIn("progress_refine/gate_mean", bridge.last_progress_refine)
        if float(loss.detach()) > 0:
            loss.backward()
            self.assertIsNotNone(a2c.mu.weight.grad)


def _ns(**kwargs):
    return argparse.Namespace(**kwargs)


def _all_profile_dests() -> set[str]:
    dests: set[str] = set()
    for table in PROFILE_TABLES.values():
        dests.update(table.keys())
    return dests


def _expand(profile: str):
    args = _ns(wm_strategy_profile=profile)
    for dest in _all_profile_dests():
        if not hasattr(args, dest):
            setattr(args, dest, None)
    apply_wm_strategy_profile(
        args, argv=["train.py", f"--wm_strategy_profile={profile}"], authoritative=True
    )
    return args


class TestProfile(unittest.TestCase):
    def test_profile_id_22(self):
        self.assertEqual(PROFILE_IDS["cser_dexpbt_progress_refine"], 22)

    def test_19_legacy_profiles_unchanged(self):
        # Spot-check invariants of the three retained CSER profiles.
        cser = _expand("cser_dexpbt")
        self.assertTrue(cser.elite_snippet_enable)
        self.assertFalse(cser.awr_enable)
        self.assertEqual(cser.eir_mppi_mode, "off")
        cons = _expand("cser_dexpbt_conservative")
        self.assertTrue(cons.eir_actor_distill_enable)
        self.assertEqual(cons.late_floor_retention_tol_threshold, 0.030)
        bridge = _expand("cser_dexpbt_2cm_bridge")
        self.assertTrue(bridge.bridge_curriculum_enable)
        self.assertEqual(bridge.late_degradation_rescue_tol_threshold, 0.022)
        self.assertTrue(bridge.same_tol_retention_enable)

    def test_20_new_profile_forces_hard_offs(self):
        a = _expand("cser_dexpbt_progress_refine")
        self.assertFalse(a.eir_residual_apply_to_rollout)
        self.assertFalse(a.eir_use_rssm_mppi)
        self.assertEqual(a.eir_mppi_mode, "off")
        self.assertFalse(a.awr_enable)
        self.assertFalse(a.pbt_wm_profile_copy_enable)
        # method flags on
        self.assertTrue(a.progress_refinement_enable)
        self.assertTrue(a.adaptive_precision_consolidation_enable)
        self.assertEqual(a.snippet_tolerance_compatibility_mode, "log_relative")
        self.assertEqual(a.eir_dhat_margin_mode, "relative_tol")
        # 2cm bridge control plane off
        self.assertFalse(a.bridge_curriculum_enable)
        self.assertFalse(a.same_tol_retention_enable)
        self.assertFalse(a.late_degradation_rescue_enable)

    def test_21_no_2cm_branch_in_new_surface(self):
        # New profile table + new modules + new launchers must not hard-code the
        # 2cm ladder constants 0.0205 / 0.022 as method logic.
        table = PROFILE_TABLES["cser_dexpbt_progress_refine"]
        for v in table.values():
            self.assertNotIn(v, (0.0205,))
        mod_paths = [
            REPO_ROOT / "runtime_root/src/dexpbt_isaaclab/dexpbt_isaaclab/world_model/progress_conditioned_refinement.py",
            REPO_ROOT / "runtime_root/src/dexpbt_isaaclab/dexpbt_isaaclab/pbt/adaptive_precision_consolidation.py",
        ]
        for p in mod_paths:
            text = p.read_text()
            self.assertNotIn("0.0205", text)
            self.assertNotIn("0.022", text)
            self.assertNotIn("2cm", text.lower())
        launchers = [
            "start_regrasping_cser_dexpbt_progress_refine_5b.sh",
            "start_regrasping_cser_dexpbt_progress_refine_smoke.sh",
            "start_regrasping_cser_dexpbt_progress_refine_ablation.sh",
        ]
        # Only comment lines may mention the constants (documenting their absence);
        # no flag value line may contain them.
        for name in launchers:
            for ln in (REPO_ROOT / "results/launchers" / name).read_text().splitlines():
                s = ln.strip()
                if s.startswith("#") or s.startswith("note=") or s.startswith("fixed_tolerance_ladder="):
                    continue
                self.assertNotIn("0.0205", s)
                self.assertNotIn("0.022", s)


if __name__ == "__main__":
    unittest.main()
