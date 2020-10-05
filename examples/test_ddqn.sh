#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Example script for periodic evaluation of a DDQN model on MICCAI2020Env as in
# https://arxiv.org/pdf/2007.10469.pdf
#
# To use it, call it from the main repo folder (path/to/active-mri-acquisition) as
#       ./examples/test_ddqn.sh <save_dir> <env> <num_parallel_episodes>
#
#   <save_dir>: Directory to save logs and model checkpoints.
#   <env>: One of "miccai" (Scenario 30L) or "miccai_extreme" (Scenario2L).
#   <num_parallel_episodes>: How many episodes the environment runs in parallel.
#
# Looks for a saved checkpoint in <save_dir> and runs it on the validation
# set. It will read the options from a pickle file stored in the same directory.

saved_checkpoint_dir=$1
env=$2
num_parallel_episodes=$3

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

python examples/test_ddqn.py \
    --training_dir "${saved_checkpoint_dir}" \
    --budget ${budget} \
    --num_parallel_episodes "${num_parallel_episodes}" \
    --device "cuda:1" \
    --seed 1 ${extreme}
