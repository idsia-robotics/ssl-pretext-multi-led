import unittest

import torch
from src.dataset.dataset import get_dataset
from torch.utils.data import DataLoader

from src.models.fcn import Model_s

class TestPretextDataset(unittest.TestCase):

    def test_supervised_flag_count(self):
        target_count = 123
        ds = get_dataset(
            "data/robomaster_ds_training.h5",
            supervised_flagging=target_count
        )
        dl = DataLoader(ds, batch_size=128)
        supervised_count = 0
        for batch in dl:
            supervised_count += batch["supervised_flag"].sum()
        self.assertEqual(supervised_count, target_count)

    def test_selector_consistency(self):
        """
        Test visibility-agnostic consistency in selector
        """
        target_count = 438
        seed = 42
        ds = get_dataset(
            "data/robomaster_ds_validation.h5",
            supervised_flagging=target_count,
            supervised_flagging_seed=seed,
        )
        dl = DataLoader(ds, batch_size=128)
        timestamps_pretext = []
        
        for batch in dl:
            timestamps_pretext.extend(batch['timestamp'][batch["supervised_flag"]].numpy().tolist())

        ds = get_dataset(
            "data/robomaster_ds_validation.h5",
            sample_count=target_count,
            sample_count_seed=seed,
            only_visible_robots=True
        )
        dl = DataLoader(ds, batch_size=128)
        timestamps_baseline = []
        
        for batch in dl:
            timestamps_baseline.extend(batch['timestamp'].numpy().tolist())
        
        timestamps_pretext = set(timestamps_pretext)
        timestamps_baseline = set(timestamps_baseline)
        self.assertEqual(timestamps_baseline, timestamps_pretext)

    def test_selector_consistency_different_seed(self):
        target_count = 438
        seed = 42
        ds = get_dataset(
            "data/robomaster_ds_validation.h5",
            supervised_flagging=target_count,
            supervised_flagging_seed=seed
        )
        dl = DataLoader(ds, batch_size=128)
        timestamps_pretext = []
        
        for batch in dl:
            timestamps_pretext.extend(batch['timestamp'][batch["supervised_flag"]].numpy().tolist())

        ds = get_dataset(
            "data/robomaster_ds_validation.h5",
            sample_count=target_count,
            sample_count_seed=seed * 2
        )
        dl = DataLoader(ds, batch_size=128)
        timestamps_baseline = []
        
        for batch in dl:
            timestamps_baseline.extend(batch['timestamp'].numpy().tolist())
        
        timestamps_pretext = set(timestamps_pretext)
        timestamps_baseline = set(timestamps_baseline)
        self.assertNotEqual(timestamps_baseline, timestamps_pretext)


if __name__ == "__main__":
    unittest.main()