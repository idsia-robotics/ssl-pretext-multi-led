import unittest
from src.dataset.dataset import get_dataset
from torch.utils.data import DataLoader

class TestBaselineDataset(unittest.TestCase):

    def test_sample_count(self):
        target_count = 399
        ds = get_dataset(
            "data/robomaster_ds_training.h5",
            sample_count=target_count
        )
        dl = DataLoader(ds, batch_size=128)
        count = 0
        supervised_count = 0
        for batch in dl:
            count += batch["image"].shape[0]
            supervised_count += batch["supervised_flag"].sum()
        self.assertEqual(count, target_count)
        self.assertEqual(supervised_count, target_count)

        

if __name__ == "__main__":
    unittest.main()