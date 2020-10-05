#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Example script for training a DDQN model on MICCAI2020Env as in
# https://arxiv.org/pdf/2007.10469.pdf
#
# To use it, call it from the main repo folder (path/to/active-mri-acquisition)
#       ./examples/train_ddqn.sh <save_dir> <env> <num_parallel_episodes> <model_type>
#
#   <save_dir>: Directory to save logs and model checkpoints.
#   <env>: One of "miccai" (Scenario 30L) or "miccai_extreme" (Scenario2L).
#   <num_parallel_episodes>: How many episodes the environment runs in parallel.
#   <model_type>: One of "ss-ddqn" (subject-specific), "ds-ddqn"
#                 (dataset-specific).
#
# If the training job gets stopped, calling this script again with the same
# <save_dir> will resume training (but the replay buffer will *not* be saved).
# As a side note, we mention that the current configuration requires about
# 100GB of memory to store the replay buffer during training, so you might need
# to reduce the value of `--mem_capacity`.
#
# By default this script does not evaluate the agent periodically. For that,
# you can take a look at test_ddqn.sh, which evaluates the saved model
# (if you pass the same <save_dir>) and saves the weights with the best
# performance on a subset of the validation set. The number of
# validation images is `--num_test_episodes` x `--num_parallel_episodes`.
#
# If you would like to interleave training and evaluation in the same job
# (e.g., due to GPU constraints), then modify this script to pass argument
#   `--dqn_test_episode_freq <NUMBER>`
# to train_ddqn.py.

set -x

saved_checkpoint_dir=$1
env=$2
num_parallel_episodes=$3
model_type=$4


if [ "${env}" == "miccai_extreme" ]; then
  budget=98
  extreme="--extreme_acc"
elif [ "${env}" == "miccai" ]; then
  budget=70
  extreme=""
else
  echo "Unrecognized environment"
  exit 1
fi

if [ "${model_type}" == "ds-ddqn" ]; then
  model_type=simple_mlp
elif [ "${model_type}" == "ss-ddqn" ]; then
  model_type=evaluator
else
  echo "Unrecognized model type"
  exit 1
fi

python examples/train_ddqn.py \
    --resume \
    --budget ${budget} \
    --num_parallel_episodes "${num_parallel_episodes}" \
    --checkpoints_dir "${saved_checkpoint_dir}" \
    --dqn_weights_path "${saved_checkpoint_dir}"/policy_checkpoint.pth \
    --dqn_model_type ${model_type} \
    --device "cuda:0" \
    --seed 1 \
    --mem_capacity 20000 \
    --reward_metric ssim \
    --mask_embedding_dim 0 \
    --dqn_batch_size 2 \
    --dqn_burn_in 1000 \
    --gamma 0.5 \
    --epsilon_start 1.0 \
    --epsilon_decay 1000000 \
    --epsilon_end 0.01 \
    --dqn_learning_rate 0.0001 \
    --num_train_steps 5000000 \
    --num_test_episodes 50 \
    --target_net_update_freq 5000 \
    --freq_dqn_checkpoint_save 50 ${extreme}
