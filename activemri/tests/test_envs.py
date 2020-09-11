import functools
import pathlib

import numpy as np

# noinspection PyUnresolvedReferences
import pytest
import torch

import activemri.envs.envs as envs
import activemri.envs.util as util
import activemri.tests.mocks as mocks


def test_import_object_from_str():
    ceil = util.import_object_from_str("math.ceil")
    assert 3 == ceil(2.5)
    det = util.import_object_from_str("numpy.linalg.det")
    assert det(np.array([[1, 0], [0, 1]])) == 1


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
class TestActiveMRIEnv:
    def test_init_from_config_dict(self):
        env = envs.ActiveMRIEnv(32, 64)
        env._init_from_config_dict(mocks.config_dict)
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
        shapes = [(1, 2) for _ in range(batch_size)]
        mask = env._mask_func(shapes, "rng")
        assert mask.shape == (batch_size, env._cfg["mask"]["args"]["size"])

    def test_init_sets_action_space(self):
        env = envs.ActiveMRIEnv(32, 64)
        for i in range(32):
            assert env.action_space.contains(i)
        assert env.action_space.n == 32

    def test_reset_and_step(self):
        # the mock environment is set up to use mocks.Reconstructor
        # and mocks.mask_function.
        # The mask and data will be tensors of size D (env._tensor_size)
        # Initial mask will be:
        #       [1 1 1 0 0 .... 0] (needs 7 actions)
        #       [1 1 0 0 0 .... 0] (needs 8 actions)
        # Ground truth is X * ones(D, D)
        # K-space is (X - 1) * ones(D D)
        # Reconstruction is K-space + Mask. So, with the initial mask we have
        # sum |reconstruction - gt| = D^2 - 3D for first element of batch,
        # and = D^2 - 2D for second element.

        def compute_score(x, y):
            return {"ssim": (x - y).abs().sum()}

        env = mocks.MRIEnv(
            batch_size=2, loops_train=1, num_train=2, score_fn=compute_score
        )
        obs, _ = env.reset()
        # env works with shape (batch, height, width, {real/img})
        assert tuple(obs["reconstruction"].shape) == (
            env.batch_size,
            env._tensor_size,
            env._tensor_size,
            2,
        )
        assert "ssim" in env._current_score

        mask_idx0_initial_active = env._cfg["mask"]["args"]["how_many"]
        mask_idx1_initial_active = mask_idx0_initial_active - 1

        def expected_score(step):
            # See explanation above, plus every steps adds one more 1 to mask.
            s = env._tensor_size
            total = s ** 2
            return 2 * (
                (total - (mask_idx0_initial_active + step) * s)
                + (total - (mask_idx1_initial_active + step) * s)
            )

        assert env._current_score["ssim"] == expected_score(0)
        prev_score = env._current_score["ssim"]
        for action in range(mask_idx0_initial_active, env._tensor_size):
            obs, reward, done, _ = env.step(action)
            assert env._current_score["ssim"] == expected_score(
                action - mask_idx1_initial_active
            )
            assert reward == env._current_score["ssim"] - prev_score
            prev_score = env._current_score["ssim"]
            if action < 9:
                assert done == [False, False]
            else:
                assert done == [True, False]
        obs, reward, done, _ = env.step(mask_idx1_initial_active)
        assert env._current_score["ssim"] == 0.0
        assert reward == -prev_score
        assert done == [True, True]

    def test_training_loop_ends(self):
        env = envs.ActiveMRIEnv(32, 64, batch_size=3)
        env._num_loops_train_data = 3
        env._init_from_config_dict(mocks.config_dict)

        num_train = 10
        tensor_size = env._cfg["mask"]["args"]["size"]

        data_init_fn = mocks.make_data_init_fn(tensor_size, num_train, 0, 0)
        env._setup_data_handlers(data_init_fn)

        seen = dict([(x, 0) for x in range(num_train)])
        for i in range(1000):
            obs, meta = env.reset()
            if not obs:
                cnt_seen = functools.reduce(lambda x, y: x + y, seen.values())
                assert cnt_seen == num_train * env._num_loops_train_data
                break
            slice_ids = meta["slice_id"]
            for id in slice_ids:
                assert id < num_train
                seen[id] = seen[id] + 1
        for i in range(num_train):
            assert seen[i] == env._num_loops_train_data

    def test_alternate_loop_modes(self):
        # This tests if the environment can change correctly between train, val, and test
        # datasets.
        num_train, num_val, num_test = 10, 7, 5
        env = mocks.MRIEnv(
            batch_size=1,
            loops_train=2,
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
        )

        # For each iteration of train data we will do a full loop over validation
        # and a partial loop over test.
        seen_train = dict([(x, 0) for x in range(num_train)])
        seen_val = dict([(x, 0) for x in range(num_val)])
        seen_test = dict([(x, 0) for x in range(num_test)])
        for i in range(1000):
            env.set_training()
            obs, meta = env.reset()
            if not obs:
                break
            for id in meta["slice_id"]:
                seen_train[id] = seen_train[id] + 1

            env.set_val()
            for j in range(num_val + 1):
                obs, meta = env.reset()
                if not obs:
                    cnt_seen = functools.reduce(lambda x, y: x + y, seen_val.values())
                    assert cnt_seen == (i + 1) * num_val
                    break
                assert j < num_val
                for id in meta["slice_id"]:
                    seen_val[id] = seen_val[id] + 1

            # With num_test - 1 we check that next call starts from 0 index again
            # even if not all images visited. One of the elements in test set should have
            # never been seen (data_handler will permute the indices so we don't know
            # which index it will be)
            env.set_test()
            for j in range(num_test - 1):
                obs, meta = env.reset()
                assert obs
                for id in meta["slice_id"]:
                    seen_test[id] = seen_test[id] + 1

        for i in range(num_train):
            assert seen_train[i] == env._num_loops_train_data

        for i in range(num_val):
            assert seen_val[i] == env._num_loops_train_data * num_train

        cnt_not_seen = 0
        for i in range(num_test):
            if seen_test[i] != 0:
                assert seen_test[i] == env._num_loops_train_data * num_train
            else:
                cnt_not_seen += 1
        assert cnt_not_seen == 1

    def test_seed(self):
        num_train = 10
        env = mocks.MRIEnv(batch_size=1, loops_train=1, num_train=num_train, seed=0)

        def get_current_order():
            order = []
            for i in range(num_train):
                obs, _ = env.reset()
                order.append(obs["reconstruction"].sum().int().item())
            return order

        order_1 = get_current_order()

        env.seed(123)
        order_2 = get_current_order()

        env.seed(0)
        order_3 = get_current_order()

        assert set(order_1) == set(order_2)
        assert any([a != b for a, b in zip(order_1, order_2)])
        assert all([a == b for a, b in zip(order_1, order_3)])


# noinspection PyProtectedMember
class TestMICCAIEnv:
    env = envs.MICCAI2020Env()

    def test_miccai_env_batch_content(self):
        for i, batch in enumerate(self.env._train_data_handler):
            # No check below for batch[1], since it's the mask and will be replaced later

            for j in [0, 1, 3, 4, 5]:
                assert isinstance(batch[j], list)
                assert len(batch[j]) == self.env.batch_size
            for batch_idx in range(self.env.batch_size):
                assert isinstance(batch[0][batch_idx], np.ndarray)
                assert batch[0][batch_idx].shape == (640, 368, 2,)  # k-space
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

    def test_reset(self):
        obs, _ = self.env.reset()
        assert len(obs) == 3
        assert "reconstruction" in obs
        assert "mask" in obs
        assert "extra_outputs" in obs
        assert obs["reconstruction"].shape == (self.env.batch_size, 640, 368, 2)
        assert obs["mask"].shape == (self.env.batch_size, 368)


# noinspection PyProtectedMember
class TestSingleCoilKneeEnv:
    env = envs.SingleCoilKneeEnv(num_cols=(368,))

    def test_singlecoilknee_env_batch_content(self):
        for i, batch in enumerate(self.env._train_data_handler):
            # No check below for batch[1], since it's the mask and will be replaced later

            kspace, _, ground_truth, attrs, fname, slice_id = batch

            for j in [0, 1, 3, 4, 5]:
                assert isinstance(batch[j], list)
                assert len(batch[j]) == self.env.batch_size
            for batch_idx in range(self.env.batch_size):
                assert isinstance(kspace[batch_idx], np.ndarray)
                assert np.all(
                    np.iscomplex(kspace[batch_idx][np.nonzero(kspace[batch_idx])])
                )
                assert kspace[batch_idx].shape == (640, 368)  # k-space
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

    def test_reset(self):
        obs, _ = self.env.reset()
        assert len(obs) == 3
        assert "reconstruction" in obs
        assert "mask" in obs
        assert "extra_outputs" in obs
        assert obs["reconstruction"].shape == (self.env.batch_size, 320, 320)
        assert obs["mask"].shape == (self.env.batch_size, 368)
