"""EliteBank actor-distill fallback alongside snippet distill."""

from __future__ import annotations

import unittest

from dexpbt_isaaclab.world_model.eir_mpc import EIRConfig, SRC_RSSM, should_use_actor_distill_fallback


def _cfg(**kwargs) -> EIRConfig:
    base = dict(
        actor_distill_enable=True,
        actor_distill_fallback_with_snippet_enable=True,
        actor_distill_fallback_min_snippet_bank_size=5000,
        actor_distill_fallback_min_rerank_changed_frac=1e-4,
        allowed_source_ids=(SRC_RSSM,),
    )
    base.update(kwargs)
    return EIRConfig(**base)


class ActorDistillFallbackWithSnippetTest(unittest.TestCase):
    def test_snippet_disabled_uses_actor_distill(self):
        use, reason = should_use_actor_distill_fallback(
            eir_cfg=_cfg(),
            use_snippet=False,
            snippet_distill_loss=0.0,
            snippet_bank_size=0.0,
            snippet_rerank_changed_frac=0.0,
        )
        self.assertTrue(use)
        self.assertEqual(reason, "snippet_disabled")

    def test_small_snippet_bank_uses_fallback(self):
        use, reason = should_use_actor_distill_fallback(
            eir_cfg=_cfg(),
            use_snippet=True,
            snippet_distill_loss=0.01,
            snippet_bank_size=1000.0,
            snippet_rerank_changed_frac=0.5,
        )
        self.assertTrue(use)
        self.assertEqual(reason, "snippet_bank_small")

    def test_zero_snippet_loss_uses_fallback(self):
        use, reason = should_use_actor_distill_fallback(
            eir_cfg=_cfg(),
            use_snippet=True,
            snippet_distill_loss=0.0,
            snippet_bank_size=10000.0,
            snippet_rerank_changed_frac=0.5,
        )
        self.assertTrue(use)
        self.assertEqual(reason, "snippet_loss_zero")

    def test_inactive_rerank_uses_fallback(self):
        use, reason = should_use_actor_distill_fallback(
            eir_cfg=_cfg(),
            use_snippet=True,
            snippet_distill_loss=0.01,
            snippet_bank_size=10000.0,
            snippet_rerank_changed_frac=0.0,
        )
        self.assertTrue(use)
        self.assertEqual(reason, "snippet_rerank_inactive")

    def test_healthy_snippet_skips_fallback(self):
        use, reason = should_use_actor_distill_fallback(
            eir_cfg=_cfg(),
            use_snippet=True,
            snippet_distill_loss=0.01,
            snippet_bank_size=10000.0,
            snippet_rerank_changed_frac=0.5,
        )
        self.assertFalse(use)
        self.assertEqual(reason, "snippet_ok_no_fallback")

    def test_actor_distill_disabled_never_runs(self):
        use, reason = should_use_actor_distill_fallback(
            eir_cfg=_cfg(actor_distill_enable=False),
            use_snippet=True,
            snippet_distill_loss=0.0,
            snippet_bank_size=0.0,
            snippet_rerank_changed_frac=0.0,
        )
        self.assertFalse(use)
        self.assertEqual(reason, "snippet_disabled")

    def test_conservative_profile_sources_exclude_rssm(self):
        cfg = _cfg(allowed_source_ids=())
        cfg = EIRConfig(
            actor_distill_enable=True,
            allowed_source_ids=tuple(),
        )
        self.assertNotIn(SRC_RSSM, cfg.allowed_source_ids)


if __name__ == "__main__":
    unittest.main()
