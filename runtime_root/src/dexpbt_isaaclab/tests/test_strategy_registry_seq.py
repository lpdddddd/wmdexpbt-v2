"""
Tests for the strategy registry extension (Phase A4).

Verifies that the world-model registry, YAML config, and PBT canonical pool
all agree on the four new sequence WM strategies.
"""

from __future__ import annotations

import unittest

from dexpbt_isaaclab.world_model.wm_trainer import (
    RSSM_FAMILY,
    SEQUENCE_STRATEGIES,
    STRATEGIES,
    TRANS_FAMILY,
    get_strategy,
)


class StrategyRegistryNamesTest(unittest.TestCase):
    NEW_NAMES = ("rssm_aux", "rssm_belief_value", "trans_aux", "trans_belief_value")

    def test_new_strategies_in_strategies_list(self):
        for name in self.NEW_NAMES:
            self.assertIn(name, STRATEGIES, f"{name} missing from STRATEGIES")

    def test_old_strategies_still_present(self):
        for name in ("none", "aux_low", "aux_mid", "value_imag_h1", "value_imag_h3"):
            self.assertIn(name, STRATEGIES)

    def test_get_strategy_returns_valid_object_for_new_names(self):
        for name in self.NEW_NAMES:
            s = get_strategy(name)
            self.assertEqual(s.name, name)
            # Sequence WM strategies must NOT be trained by the legacy
            # single-step trainer (SeqWMTrainer takes over).
            self.assertFalse(s.trains_wm,
                             f"{name} should be trains_wm=False for legacy trainer")
            self.assertFalse(s.use_value_imag)
            self.assertEqual(s.lambda_wm, 0.0)

    def test_get_strategy_unknown_raises(self):
        with self.assertRaises(ValueError):
            get_strategy("rssm_does_not_exist")

    def test_family_sets(self):
        self.assertEqual(RSSM_FAMILY, {"rssm_aux", "rssm_belief_value"})
        self.assertEqual(TRANS_FAMILY, {"trans_aux", "trans_belief_value"})
        self.assertEqual(SEQUENCE_STRATEGIES, RSSM_FAMILY | TRANS_FAMILY)
        self.assertNotIn("value_imag_h1", SEQUENCE_STRATEGIES)
        self.assertNotIn("aux_mid", SEQUENCE_STRATEGIES)


class StrategyYamlConsistencyTest(unittest.TestCase):
    """The YAML defaults must match the python defaults for the new entries."""

    def test_yaml_loads_each_seq_strategy(self):
        # YAML loading is exercised inside get_strategy when the python
        # defaults table doesn't have a key. Even though we DO list the new
        # ones in defaults, we still parse YAML for completeness.
        from importlib import resources
        import yaml
        cfg_file = resources.files("dexpbt_isaaclab.agents").joinpath(
            "wm_strategies.yaml"
        )
        with open(str(cfg_file)) as f:
            data = yaml.safe_load(f)
        for name in ("rssm_aux", "rssm_belief_value",
                     "trans_aux", "trans_belief_value"):
            self.assertIn(name, data, f"{name} missing from wm_strategies.yaml")
            entry = data[name]
            self.assertFalse(entry["trains_wm"])
            self.assertEqual(float(entry["lambda_wm"]), 0.0)


class CanonicalPoolTest(unittest.TestCase):
    """Re-create just the canonical_pool list to verify pbt_observer edit."""

    def test_canonical_pool_contains_new_names_and_old(self):
        # We import the module's source and grep — avoids constructing a
        # fully wired DexPBTPbtAlgoObserver (which requires the full algo
        # info dict). This catches accidental name regressions.
        from importlib import resources
        src = resources.files("dexpbt_isaaclab.pbt").joinpath(
            "pbt_observer.py"
        ).read_text()
        for name in (
            "none", "aux_low", "aux_mid", "value_imag_h1", "value_imag_h3",
            "rssm_aux", "rssm_belief_value",
            "trans_aux", "trans_belief_value",
        ):
            self.assertIn(f'"{name}"', src,
                          f"{name} missing from pbt_observer canonical pool")

    def test_value_imag_h3_aux_low_still_listed(self):
        """They are CANONICAL (so legacy runs still validate), but the
        forbidden-pool gate in launchers + report generator filters them
        out of new sequence runs. We test launcher gating in a separate
        end-to-end test once Phase A4 launchers exist."""
        from importlib import resources
        src = resources.files("dexpbt_isaaclab.pbt").joinpath(
            "pbt_observer.py"
        ).read_text()
        self.assertIn('"value_imag_h3"', src)
        self.assertIn('"aux_low"', src)


if __name__ == "__main__":
    unittest.main()
