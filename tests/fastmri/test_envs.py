# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest  # noqa: F401

import activemri.envs.envs as envs


class TestMICCAIEnv:
    env = envs.MICCAI2020Env()

    def test_miccai_env_batch_content(self):
        for i, batch in enumerate(self.env._train_data_handler):
            # No check below for batch[1], since it's the mask and will be replaced later

            for j in [0, 1, 3, 4, 5]:
                assert isinstance(batch[j], list)
                assert len(batch[j]) == self.env.num_parallel_episodes
            for batch_idx in range(self.env.num_parallel_episodes):
                assert isinstance(batch[0][batch_idx], np.ndarray)
                assert batch[0][batch_idx].shape == (
                    640,
                    368,
                    2,
                )  # k-space
                assert isinstance(batch[2][batch_idx], np.ndarray)
                assert batch[2][batch_idx].shape == (640, 368, 2)  # ground truth image

                # data.attrs
                assert len(batch[3][batch_idx]) == 4
                for key in ["norm", "max", "patient_id", "acquisition"]:
                    assert key in batch[3][batch_idx]
                # file name
                assert isinstance(batch[4][batch_idx], str)
                # slice_id
                assert isinstance(batch[5][batch_idx], int)
            if i == 10:
                break

    def test_miccai_reset(self):
        obs, _ = self.env.reset()
        assert len(obs) == 3
        assert "reconstruction" in obs
        assert "mask" in obs
        assert "extra_outputs" in obs
        assert obs["reconstruction"].shape == (
            self.env.num_parallel_episodes,
            640,
            368,
            2,
        )
        assert obs["mask"].shape == (self.env.num_parallel_episodes, 368)


class TestSingleCoilKneeEnv:
    env = envs.SingleCoilKneeEnv()

    def test_singlecoil_knee_env_batch_content(self):
        for i, batch in enumerate(self.env._train_data_handler):
            # No check below for batch[1], since it's the mask and will be replaced later

            kspace, _, ground_truth, attrs, fname, slice_id = batch

            for j in [0, 1, 3, 4, 5]:
                assert isinstance(batch[j], list)
                assert len(batch[j]) == self.env.num_parallel_episodes
            for batch_idx in range(self.env.num_parallel_episodes):
                assert isinstance(kspace[batch_idx], np.ndarray)
                assert np.all(
                    np.iscomplex(kspace[batch_idx][np.nonzero(kspace[batch_idx])])
                )
                assert kspace[batch_idx].shape in [(640, 368), (640, 372)]  # k-space
                assert isinstance(ground_truth[batch_idx], np.ndarray)
                assert not np.any(np.iscomplex(ground_truth[batch_idx]))
                assert ground_truth[batch_idx].shape == (320, 320)  # ground_truth

                # data.attrs
                assert len(attrs[batch_idx]) == 8
                for key in [
                    "acquisition",
                    "max",
                    "norm",
                    "patient_id",
                    "padding_left",
                    "padding_right",
                    "encoding_size",
                    "recon_size",
                ]:
                    assert key in attrs[batch_idx]
                # file name
                assert isinstance(fname[batch_idx], str)
                # slice_id
                assert isinstance(slice_id[batch_idx], int)
            if i == 10:
                break

    def test_singlecoil_knee_reset(self):
        obs, _ = self.env.reset()
        assert len(obs) == 3
        assert "reconstruction" in obs
        assert "mask" in obs
        assert "extra_outputs" in obs
        assert obs["reconstruction"].shape == (self.env.num_parallel_episodes, 320, 320)
        assert obs["mask"].shape in [
            (self.env.num_parallel_episodes, 368),
            (self.env.num_parallel_episodes, 372),
        ]
