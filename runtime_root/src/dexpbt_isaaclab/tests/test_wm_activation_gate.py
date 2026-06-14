"""Unit tests for the PBT-WM Soft Activation Gate v1."""

from __future__ import annotations

import unittest

from dexpbt_isaaclab.world_model.wm_activation_gate import (
    WMActivationConfig,
    clamp01,
    compute_wm_activation,
    soft_tol_weight,
    source_weight,
    tol_bin,
)


class TestSoftTolWeight(unittest.TestCase):
    def test_soft_tol_weight_monotonic(self):
        # tol smaller -> larger weight
        kw = dict(tol_start=0.060, tol_full=0.040, min_w=0.0, max_w=1.0)
        w_high = soft_tol_weight(0.058, **kw)
        w_mid = soft_tol_weight(0.050, **kw)
        w_low = soft_tol_weight(0.042, **kw)
        self.assertLess(w_high, w_mid)
        self.assertLess(w_mid, w_low)
        # tol >= start -> min
        self.assertEqual(soft_tol_weight(0.060, **kw), 0.0)
        self.assertEqual(soft_tol_weight(0.10, **kw), 0.0)
        # tol <= full -> max
        self.assertEqual(soft_tol_weight(0.040, **kw), 1.0)
        self.assertEqual(soft_tol_weight(0.010, **kw), 1.0)

    def test_soft_tol_weight_degenerate_schedule(self):
        # start <= full -> hard-ish fallback
        self.assertEqual(soft_tol_weight(0.05, tol_start=0.04, tol_full=0.04), 0.0)
        self.assertEqual(soft_tol_weight(0.03, tol_start=0.04, tol_full=0.04), 1.0)

    def test_tol_bin_boundaries(self):
        self.assertEqual(tol_bin(0.07), "gt_006")
        self.assertEqual(tol_bin(0.055), "005_006")
        self.assertEqual(tol_bin(0.045), "004_005")
        self.assertEqual(tol_bin(0.04), "le_004")
        self.assertEqual(tol_bin(0.01), "le_004")

    def test_clamp01(self):
        self.assertEqual(clamp01(-1.0), 0.0)
        self.assertEqual(clamp01(2.0), 1.0)
        self.assertEqual(clamp01(float("nan")), 0.0)


class TestComputeActivation(unittest.TestCase):
    def _cfg(self, **kw):
        base = dict(
            gate_mode="soft_tol",
            tol_start=0.060,
            tol_full=0.040,
            start_after_frames=500_000_000,
            ramp_frames=300_000_000,
            nsucc_ref=20.0,
            margin_norm=0.05,
            uncertainty_threshold=2.0,
        )
        base.update(kw)
        return WMActivationConfig(**base)

    def test_compute_activation_before_start_zero(self):
        cfg = self._cfg()
        a = compute_wm_activation(
            cfg=cfg, frames=100_000_000, tol=0.045, margin=0.1,
            uncertainty=0.1, source="rssm", nsucc_mean=20.0,
        )
        self.assertEqual(a.time_weight, 0.0)
        self.assertEqual(a.weight, 0.0)
        self.assertFalse(a.active)
        self.assertEqual(a.reason, "before_start")

    def test_compute_activation_confidence_only_ignores_tol(self):
        cfg = self._cfg(gate_mode="confidence_only", start_after_frames=0)
        # high tol should still produce nonzero weight when eval valid + low unc
        a = compute_wm_activation(
            cfg=cfg, frames=10_000_000, tol=0.20, margin=0.1,
            uncertainty=0.1, source="rssm",
        )
        self.assertGreater(a.weight, 0.0)
        self.assertEqual(a.tol_weight, 1.0)
        self.assertTrue(a.active)

    def test_compute_activation_shrink_candidate_requires_token(self):
        cfg = self._cfg(gate_mode="shrink_candidate", start_after_frames=0)
        off = compute_wm_activation(
            cfg=cfg, frames=10_000_000, tol=0.05, margin=0.1,
            uncertainty=0.1, source="rssm", shrink_candidate=False,
        )
        self.assertEqual(off.weight, 0.0)
        on = compute_wm_activation(
            cfg=cfg, frames=10_000_000, tol=0.05, margin=0.1,
            uncertainty=0.1, source="rssm", shrink_candidate=True,
        )
        self.assertGreater(on.weight, 0.0)

    def test_source_weights_block_actor_and_gaussian(self):
        cfg = self._cfg(start_after_frames=0)
        for src in ("actor", "gaussian"):
            a = compute_wm_activation(
                cfg=cfg, frames=10_000_000_000, tol=0.040, margin=0.1,
                uncertainty=0.1, source=src, nsucc_mean=20.0,
            )
            self.assertEqual(a.source_weight, 0.0)
            self.assertEqual(a.weight, 0.0)
        self.assertEqual(source_weight("actor", cfg), 0.0)
        self.assertEqual(source_weight("gaussian", cfg), 0.0)
        self.assertGreater(source_weight("elite", cfg), 0.0)

    def test_rssm_uncertainty_blocks_high_uncertainty(self):
        cfg = self._cfg(start_after_frames=0, uncertainty_threshold=2.0)
        # uncertainty >= threshold -> unc weight 0
        a = compute_wm_activation(
            cfg=cfg, frames=10_000_000_000, tol=0.040, margin=0.1,
            uncertainty=2.0, source="rssm", nsucc_mean=20.0,
        )
        self.assertEqual(a.unc_weight, 0.0)
        self.assertEqual(a.weight, 0.0)
        # NaN uncertainty also blocks
        b = compute_wm_activation(
            cfg=cfg, frames=10_000_000_000, tol=0.040, margin=0.1,
            uncertainty=float("nan"), source="rssm", nsucc_mean=20.0,
        )
        self.assertEqual(b.unc_weight, 0.0)

    def test_disabled_and_eval_invalid(self):
        cfg = self._cfg(enable=False, start_after_frames=0)
        a = compute_wm_activation(cfg=cfg, frames=10_000_000_000, tol=0.04)
        self.assertEqual(a.weight, 0.0)
        self.assertEqual(a.reason, "disabled")
        cfg2 = self._cfg(start_after_frames=0)
        b = compute_wm_activation(
            cfg=cfg2, frames=10_000_000_000, tol=0.04, eval_valid=False
        )
        self.assertEqual(b.weight, 0.0)
        self.assertEqual(b.reason, "invalid_eval")


if __name__ == "__main__":
    unittest.main()
