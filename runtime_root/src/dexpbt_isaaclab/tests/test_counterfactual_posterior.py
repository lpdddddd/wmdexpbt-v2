import json
import tempfile
import unittest
from pathlib import Path

from dexpbt_isaaclab.world_model.wm_posterior_selector import SharedPosteriorSelector


class CounterfactualPosteriorTest(unittest.TestCase):
    def test_posterior_update_and_none_fixed_mean(self):
        with tempfile.TemporaryDirectory() as td:
            shared_path = Path(td) / "wm_posterior_latest.json"

            writer = SharedPosteriorSelector(shared_path=shared_path)
            writer.load_or_init(policy_idx=0)
            writer.update_and_save(
                delta_w={
                    "none": 0.0,
                    "aux_low": 0.2,
                    "aux_mid": -0.1,
                    "value_imag_h1": 0.3,
                    "value_imag_h3": 0.4,
                },
                step=123,
            )

            data = json.loads(shared_path.read_text())
            self.assertIn("none", data)
            self.assertEqual(float(data["none"]["mean"]), 0.0, "none mean must stay fixed at 0")
            self.assertGreaterEqual(int(data["value_imag_h3"]["n"]), 2)

            # Non-writer worker should load shared posterior but not save by default flow.
            non_writer = SharedPosteriorSelector(shared_path=shared_path)
            non_writer.load_or_init(policy_idx=1)
            non_writer.update_batch({"aux_low": 0.1, "none": 0.0}, step=124)
            # File should remain unchanged unless save() is explicitly called.
            data_after = json.loads(shared_path.read_text())
            self.assertEqual(data_after, data)

            selected = writer.sample()
            self.assertIn(selected, writer.strategies)


if __name__ == "__main__":
    unittest.main()
