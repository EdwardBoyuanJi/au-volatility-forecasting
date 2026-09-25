import unittest

from au_rv.models.control_experiment import all_block_subsets, control_blocks


class ControlExperimentTests(unittest.TestCase):
    def test_eight_blocks_generate_all_256_subsets(self):
        blocks = list(control_blocks(20))
        subsets = all_block_subsets(blocks)
        self.assertEqual(len(blocks), 8)
        self.assertEqual(len(subsets), 256)
        self.assertIn((), subsets)
        self.assertIn(tuple(blocks), subsets)

    def test_event_block_is_horizon_specific(self):
        self.assertIn("cpi_count_40d", control_blocks(40)["events"])


if __name__ == "__main__":
    unittest.main()
