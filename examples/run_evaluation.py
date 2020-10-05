# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import pickle

from typing import cast

import numpy as np
import torch

import activemri.baselines as baselines
import activemri.envs as envs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--budget", type=int, default=100, help="How many k-space columns to acquire."
    )
    parser.add_argument(
        "--num_parallel_episodes",
        type=int,
        default=1,
        help="The number of episodes the environment runs in parallel",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="How many batches of episodes to run in total.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        choices=[
            "random",
            "random-lb",
            "lowtohigh",
            "evaluator",
            "ds-ddqn",
            "ss-ddqn",
            "oracle",
        ],
        help="The algorithm to evaluate.",
    )
    parser.add_argument(
        "--evaluator_path",
        type=str,
        default=None,
        help="Path to checkpoint for evalutor network.",
    )
    parser.add_argument(
        "--baseline_device",
        type=str,
        default="cpu",
        help="Which torch device to use for the baseline (if 'evaluator' or '*ddqn').",
    )

    parser.add_argument(
        "--dqn_checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint for the DDQN agent.",
    )
    parser.add_argument("--legacy_model", action="store_true")

    parser.add_argument(
        "--oracle_num_samples",
        type=int,
        default=20,
        help="If using the one step greedy oracle, how many actions to sample each step.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where results will be stored.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the environment.")

    parser.add_argument("--env", choices=["miccai", "miccai_extreme"])

    args = parser.parse_args()

    extreme = "_extreme" in args.env
    env = envs.MICCAI2020Env(args.num_parallel_episodes, args.budget, extreme=extreme)

    policy: baselines.Policy = None
    if args.baseline == "random":
        policy = baselines.RandomPolicy()
    if args.baseline == "random-lb":
        policy = baselines.RandomLowBiasPolicy(acceleration=3.0, centered=False)
    if args.baseline == "lowtohigh":
        policy = baselines.LowestIndexPolicy(alternate_sides=True, centered=False)
    if args.baseline == "evaluator":
        policy = baselines.CVPR19Evaluator(
            args.evaluator_path,
            torch.device(args.baseline_device),
            add_mask=True,
        )
    if args.baseline == "oracle":
        policy = baselines.OneStepGreedyOracle(
            env, "ssim", num_samples=args.oracle_num_samples
        )
    if "ddqn" in args.baseline:
        checkpoint_path = os.path.join(
            args.dqn_checkpoint_path, "evaluation", "policy_best.pt"
        )
        checkpoint = torch.load(args.dqn_checkpoint_path)
        options = checkpoint["options"]
        if "miccai" in args.env:
            initial_num_lines = 1 if "extreme" in args.env else 15
            if args.legacy_model:
                options.legacy_offset = initial_num_lines
        policy = cast(baselines.DDQN, policy)
        policy = baselines.DDQN(args.baseline_device, None, options)
        policy.load_state_dict(checkpoint["dqn_weights"])
    all_scores, all_img_idx = baselines.evaluate(
        env, policy, args.num_episodes, args.seed, "test", verbose=True
    )

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "scores.npy"), all_scores)
    with open(os.path.join(args.output_dir, "img_ids.pkl"), "wb") as f:
        pickle.dump(all_img_idx, f)
