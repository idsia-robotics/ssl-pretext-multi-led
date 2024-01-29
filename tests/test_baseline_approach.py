import unittest

import numpy as np
from src.dataset.dataset import get_dataset
from torch.utils.data import DataLoader
from src.models.fcn import Model_s

class TestBaselineDataset(unittest.TestCase):

    def test_sample_count(self):
        target_count = 399
        ds = get_dataset(
            "data/robomaster_ds_validation.h5",
            supervised_flagging=target_count
        )
        dl = DataLoader(ds, batch_size=128)
        count = 0
        supervised_count = 0
        for batch in dl:
            count += batch["image"].shape[0]
            supervised_count += batch["supervised_flag"].sum()
        self.assertEqual(count, len(ds))
        self.assertEqual(supervised_count, target_count)


    def test_subset_disjunction(self):
        target_count = 399
        ds = get_dataset(
            "data/robomaster_ds_validation.h5",
            supervised_flagging=target_count,
            supervised_flagging_seed=0
        )
        dl = DataLoader(ds, batch_size=128)
        supervised_ts = []
        unsupervised_ts = []
        visible_and_supervised = []

        for batch in dl:
            supervised_flag = batch["supervised_flag"]
            # breakpoint()
            visible_and_supervised.extend(batch["supervised_flag"] | ~batch["robot_visible"])
            supervised_ts.extend(batch["timestamp"][supervised_flag])
            unsupervised_ts.extend(batch["timestamp"][~supervised_flag])

        supervised_ts = set(supervised_ts)
        unsupervised_ts = set(unsupervised_ts)

        self.assertTrue(len(supervised_ts), target_count)
        self.assertTrue(np.stack(visible_and_supervised).all())
        self.assertTrue(supervised_ts.isdisjoint(unsupervised_ts))

    def test_loss_masking(self):
        model = Model_s(task='pose_and_led')
        target_count = 1337
        ds = get_dataset(
            "data/robomaster_ds_validation.h5",
            supervised_flagging=target_count
        )
        dl = DataLoader(ds, batch_size=128)
        supervised_ts = []
        unsupervised_ts = []

        for batch in dl:
            supervised_flag = batch["supervised_flag"]
            supervised_ts.extend(batch["timestamp"][supervised_flag])
            unsupervised_ts.extend(batch["timestamp"][~supervised_flag])
            out = model(batch['image'])
            p_loss, d_loss, o_loss, led_loss, _ = model.loss(batch, out)

            self.assertTrue(p_loss[~supervised_flag].sum() == 0)
            self.assertTrue(d_loss[~supervised_flag].sum() == 0)
            self.assertTrue(o_loss[~supervised_flag].sum() == 0)
            self.assertTrue(led_loss[supervised_flag].sum() == 0)

            self.assertFalse(p_loss[supervised_flag].sum() == 0)
            self.assertFalse(d_loss[supervised_flag].sum() == 0)
            self.assertFalse(o_loss[supervised_flag].sum() == 0)
            self.assertFalse(led_loss[~supervised_flag].sum() == 0)
            




        

if __name__ == "__main__":
    unittest.main()