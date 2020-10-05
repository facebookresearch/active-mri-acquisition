# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch

import activemri.baselines as mri_baselines
import activemri.envs as envs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--num_parallel_episodes", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extreme_acc", action="store_true")

    parser.add_argument("--checkpoints_dir", type=str, default=None)
    parser.add_argument("--mem_capacity", type=int, default=1000)
    parser.add_argument(
        "--dqn_model_type",
        type=str,
        choices=["simple_mlp", "evaluator"],
        default="evaluator",
    )
    parser.add_argument(
        "--reward_metric",
        type=str,
        choices=["mse", "ssim", "nmse", "psnr"],
        default="ssim",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mask_embedding_dim", type=int, default=0)
    parser.add_argument("--dqn_batch_size", type=int, default=2)
    parser.add_argument("--dqn_burn_in", type=int, default=100)
    parser.add_argument("--dqn_normalize", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_decay", type=int, default=10000)
    parser.add_argument("--epsilon_end", type=float, default=0.001)
    parser.add_argument("--dqn_learning_rate", type=float, default=0.001)
    parser.add_argument("--num_train_steps", type=int, default=1000)
    parser.add_argument("--num_test_episodes", type=int, default=2)
    parser.add_argument("--dqn_only_test", action="store_true")
    parser.add_argument("--dqn_weights_path", type=str, default=None)
    parser.add_argument("--dqn_test_episode_freq", type=int, default=None)
    parser.add_argument("--target_net_update_freq", type=int, default=5000)
    parser.add_argument("--freq_dqn_checkpoint_save", type=int, default=1000)

    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    env = envs.MICCAI2020Env(
        args.num_parallel_episodes,
        args.budget,
        obs_includes_padding=args.dqn_model_type == "evaluator",
        extreme=args.extreme_acc,
    )
    env.seed(args.seed)
    policy = mri_baselines.DDQNTrainer(args, env, torch.device(args.device))
    policy()
