import unittest

from dexpbt_isaaclab.pbt.pbt_observer import _mutate_params
from dexpbt_isaaclab.world_model.wm_trainer import get_strategy


class BaselineCleanTest(unittest.TestCase):
    def test_baseline_wm_keys_not_mutated(self):
        params = {
            "gamma": 0.99,
            "mini_epochs": 4,
            "enable_wm": False,
            "wm_strategy": "none",
            "wm_ensemble_size": 3,
            "enable_counterfactual": False,
            "cf_interval_gens": 5,
            "cf_steps": 5_000_000,
        }
        for _ in range(32):
            mutated = _mutate_params(params, mutation_rate=1.0, change_min=1.1, change_max=1.5)
            self.assertFalse(mutated["enable_wm"])
            self.assertEqual(mutated["wm_strategy"], "none")
            self.assertEqual(mutated["wm_ensemble_size"], 3)
            self.assertFalse(mutated["enable_counterfactual"])
            self.assertEqual(mutated["cf_interval_gens"], 5)
            self.assertEqual(mutated["cf_steps"], 5_000_000)

    def test_none_strategy_has_no_value_imag(self):
        strategy = get_strategy("none")
        self.assertFalse(strategy.trains_wm)
        self.assertFalse(strategy.use_value_imag)


if __name__ == "__main__":
    unittest.main()
