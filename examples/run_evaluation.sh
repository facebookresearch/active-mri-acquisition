#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This script can be used to reproduce the results shown in our MICCAI'20 paper
# (https://arxiv.org/pdf/2007.10469.pdf)
# To use it, call it from the main repo folder (path/to/active-mri-acquisition) as
#       ./examples/run_evaluation.sh <args>
#
# To run the "evaluator" and "*ddqn" baselines, you need to download our trained checkpoints,
# as explained in the Documentation (https://facebookresearch.github.io/active-mri-acquisition/misc.html).
# Then pass the directory where you saved the model as the first argument in the commands below.
#
# For Scenario30L, run as
#     ./examples/run_evaluation.sh <saved_models_dir> <outputs_root> <baseline> miccai <num_parallel_episodes> 1851 70
#
# For Scenario2L, run as
#     ./examples/run_evaluation.sh <saved_models_dir> <outputs_root> <baseline> miccai_extreme <num_parallel_episodes> 1851 98
#
# Baseline options are: random, random-lb, lowtohigh, evaluator, ss-ddqn, oracle.
#
# The script will produce a file <output_dir>/<baseline>/scores.npy with the scores
# (mse, ssim, nmse, psnr) of all episodes and all time steps.
#
# Note1: The open sourced version of the oracle baseline does not used parallelization
# and is thus very slow to run.
#
# Note2: Our pretrained DDQN models used to report results in the paper were trained with
# an older version of the environment. To be able to use these models please enable
# "legacy_model" flag.


saved_models_dir=$1
output_dir=$2/${env}/${baseline}
baseline=$3
env=$4
num_parallel_episodes=$5
num_episodes=$6
budget=$7

if [ "${env}" == "miccai_extreme" ]; then
  model_type=extreme
else
  model_type=normal
fi

DQN_CKPT=${saved_models_dir}/miccai2020_${baseline}_ssim_${model_type}.pth

python examples/run_evaluation.py \
      --num_parallel_episodes "${num_parallel_episodes}" \
      --baseline "${baseline}" \
      --budget "${budget}" \
      --num_episodes "${num_episodes}" \
      --evaluator_path "${SAVED_MODELS}/miccai2020_evaluator_raw_${model_type}.pth" \
      --dqn_checkpoint_path "${DQN_CKPT}" \
      --baseline_device "cuda:0" \
      --seed 0 \
      --legacy_model \
      --env "${env}" \
      --output_dir "${output_dir}"



