import json
import pathlib

import numpy as np

# noinspection PyUnresolvedReferences
import pytest
import torch

import activemri.envs.envs as envs
import activemri.envs.mask_functions as masks
import activemri.envs.util as util
import activemri.tests.mocks as mocks


def test_import_object_from_str():
    ceil = util.import_object_from_str("math.ceil")
    assert 3 == ceil(2.5)
    det = util.import_object_from_str("numpy.linalg.det")
    assert det(np.array([[1, 0], [0, 1]])) == 1


def test_update_masks_from_indices():
    mask_1 = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.uint8)
    mask_2 = torch.tensor([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]], dtype=torch.uint8)
    mask = torch.stack([mask_1, mask_2])
    mask = masks.update_masks_from_indices(mask, np.array([2, 0]))
    assert mask.shape == torch.Size([2, 3, 4])

    expected = torch.tensor(
        [[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]], dtype=torch.uint8
    ).repeat(2, 1, 1)
    assert (mask - expected).sum().item() == 0


def test_random_cyclic_sampler_default_order():
    alist = [0, 1, 2]
    sampler = envs.CyclicSampler(alist, None, loops=10)
    cnt = 0
    for i, x in enumerate(sampler):
        assert alist[x] == i % 3
        cnt += 1
    assert cnt == 30


def test_random_cyclic_sampler_default_given_order():
    alist = [1, 2, 0]
    sampler = envs.CyclicSampler(alist, order=[2, 0, 1], loops=10)
    cnt = 0
    for i, x in enumerate(sampler):
        assert alist[x] == i % 3
        cnt += 1
    assert cnt == 30


def test_data_handler():
    data = list(range(10))
    batch_size = 2
    loops = 3
    handler = envs.DataHandler(data, None, batch_size=batch_size, loops=loops)
    cnt = dict([(x, 0) for x in data])
    for x in handler:
        assert len(x) == batch_size
        for t in x:
            v = t.item()
            cnt[v] = cnt[v] + 1
    for x in cnt:
        assert cnt[x] == loops


# noinspection PyProtectedMember
class TestMRIEnvs:

    mock_config_dict = json.loads(mocks.cfg_json_str)

    def test_init_from_config_dict(self):
        env = envs.ActiveMRIEnv(32, 64)
        env._init_from_config_dict(self.mock_config_dict)
        assert env.reward_metric == "ssim"
        assert type(env._reconstructor) == mocks.Reconstructor
        assert env._reconstructor.option1 == 1
        assert env._reconstructor.option2 == 0.5
        assert env._reconstructor.option3 == "dummy"
        assert env._reconstructor.option4
        assert env._reconstructor.weights == "init"
        assert env._reconstructor._eval
        assert env._reconstructor.device == torch.device("cpu")
        assert env._transform("x", "m") == ("x", "m")

        batch_size = 3
        mask = env._mask_func(batch_size, "rng")
        assert mask.shape == (batch_size, mocks.Dataset.size)

    def test_init_sets_action_space(self):
        env = envs.ActiveMRIEnv(32, 64)
        for i in range(32):
            assert env.action_space.contains(i)
        assert env.action_space.n == 32

    def test_reset_and_step(self):
        env = envs.ActiveMRIEnv(32, 64, batch_size=2)
        env._init_from_config_dict(self.mock_config_dict)

        def compute_score(x, y):
            return {"ssim": (x - y).abs().sum()}

        env._compute_score_given_tensors = compute_score

        data = mocks.Dataset()
        handler = envs.DataHandler(data, None, batch_size=env._batch_size, loops=1)
        env._train_data_handler = handler

        obs, _ = env.reset()
        assert tuple(obs["reconstruction"].shape) == (
            env._batch_size,
            data.size,
            data.size,
            2,
        )
        assert "ssim" in env._current_score

        def expected_score(step):
            s = mocks.Dataset.size
            total = s ** 2
            return 2 * ((total - (3 + step) * s) + (total - (2 + step) * s))

        assert env._current_score["ssim"] == expected_score(0)
        prev_score = env._current_score["ssim"]
        for action in range(3, 10):
            obs, reward, done, _ = env.step(action)
            assert env._current_score["ssim"] == expected_score(action - 2)
            assert reward == env._current_score["ssim"] - prev_score
            prev_score = env._current_score["ssim"]
            if action < 9:
                assert done == [False, False]
            else:
                assert done == [True, False]
        obs, reward, done, _ = env.step(2)
        assert env._current_score["ssim"] == 0.0
        assert reward == -prev_score
        assert done == [True, True]


# noinspection PyProtectedMember
class TestSingleCoilRawEnv:
    env = envs.SingleCoilKneeRAWEnv()

    def test_singlecoil_raw_env_batch_content(self):
        for i, batch in enumerate(self.env._train_data_handler):
            # No check for batch[1], since it's the mask and will be replaced later
            assert batch[0].shape == (
                self.env._batch_size,
                640,
                368,
                2,
            )  # reconstruction input
            assert batch[2].shape == (self.env._batch_size, 640, 368, 2)  # target image
            for j in range(3, 6):
                assert len(batch[j]) == self.env._batch_size
            for l in range(self.env._batch_size):
                # data.attrs
                assert len(batch[3][l]) == 4
                for key in ["norm", "max", "patient_id", "acquisition"]:
                    assert key in batch[3][l]
                # file name
                assert isinstance(batch[4][l], pathlib.Path)
                # slice_id
                assert isinstance(batch[5][l], int)
            if i == 1:
                break

    def test_reset(self):
        obs, _ = self.env.reset()
        assert len(obs) == 3
        assert "reconstruction" in obs
        assert "mask" in obs
        assert "extra_outputs" in obs
        assert obs["reconstruction"].shape == (self.env._batch_size, 640, 368, 2)
        assert obs["mask"].shape == (self.env._batch_size, 368)
