# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # noqa: F401
import torch

import activemri.baselines as baselines


def test_random():
    policy = baselines.RandomPolicy()

    bs = 4
    mask = torch.zeros(bs, 10)
    mask[:, :3] = 1
    mask[0, :7] = 1
    obs = {"mask": mask}

    steps = 5
    for i in range(steps):
        action = policy(obs)
        assert len(action) == bs
        for j in range(bs):
            if j > 0 or (j == 0 and i < 3):
                assert obs["mask"][j, action[j]] == 0
            obs["mask"][j, action[j]] = 1

    assert obs["mask"].sum().item() == 34


def test_low_to_high_no_alternate():
    policy = baselines.LowestIndexPolicy(alternate_sides=False, centered=False)

    mask = torch.zeros(2, 10)
    mask[0, 0::2] = 1
    mask[1, 1::2] = 1
    obs = {"mask": mask}

    for i in range(5):
        action = policy(obs)
        assert len(action) == 2
        assert action[0] == 2 * i + 1
        assert action[1] == 2 * i
        obs["mask"][:, action] = 1

    assert obs["mask"].sum().item() == 20


def test_low_to_high_alternate():
    policy = baselines.LowestIndexPolicy(alternate_sides=True, centered=False)

    mask = torch.zeros(2, 10)
    mask[0, 0::2] = 1
    mask[1, 1::2] = 1
    obs = {"mask": mask}

    order = [[1, 9, 3, 7, 5], [0, 8, 2, 6, 4]]
    for i in range(5):
        action = policy(obs)
        assert len(action) == 2
        assert action[0] == order[0][i]
        assert action[1] == order[1][i]
        obs["mask"][:, action] = 1

    assert obs["mask"].sum().item() == 20


def test_low_to_high_alternate_centered():
    policy = baselines.LowestIndexPolicy(alternate_sides=True, centered=True)

    mask = torch.zeros(2, 10)
    mask[0, 0::2] = 1
    mask[1, 1::2] = 1
    obs = {"mask": mask}

    order = [[5, 3, 7, 1, 9], [6, 4, 8, 2, 0]]
    for i in range(5):
        action = policy(obs)
        assert len(action) == 2
        assert action[0] == order[0][i]
        assert action[1] == order[1][i]
        obs["mask"][:, action] = 1

    assert obs["mask"].sum().item() == 20
