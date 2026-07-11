"""PBT tier count splits for ranking."""

import unittest

from dexpbt_isaaclab.pbt.pbt_observer import _compute_tier_counts


class TierCountsTest(unittest.TestCase):
    def test_six_policy_split(self):
        self.assertEqual(_compute_tier_counts(6), (1, 3, 2))

    def test_eight_policy_split(self):
        self.assertEqual(_compute_tier_counts(8), (2, 3, 3))


if __name__ == "__main__":
    unittest.main()
