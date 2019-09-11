#!/bin/bash

CHECKPOINTS_BASE=/checkpoint/lep/active_acq
MODELS_DIR=${CHECKPOINTS_BASE}/all_reconstructors
MODEL_TYPE=symmetric_basic_rnl

SRC_DIR=/private/home/lep/code/versions/Active_Acquisition/tune_fourier_$(date +%Y%m%d_%H.%M.%S)

mkdir -p ${SRC_DIR}
cp -r /private/home/lep/code/Active_Acquisition/* ${SRC_DIR}

cd ${SRC_DIR}

python tune_dqn.py --dataroot KNEE \
    --reconstructor_dir ${MODELS_DIR}/${MODEL_TYPE}\
    --evaluator_dir ${MODELS_DIR}/${MODEL_TYPE}/evaluator \
    --checkpoints_dir ${CHECKPOINTS_BASE}/dqn_tuning/budget11_randomsearch/${MODEL_TYPE} \
    --test_set val \
    --num_test_images 100 \
    --use_reconstructions \
    --freq_save_test_stats 100000 \
    --dqn_test_episode_freq 20 \
    --sequential_images \
    --initial_num_lines 5 \
    --obs_type fourier_space \
    --dqn_resume \
    --rl_env_train_no_seed \
    --budget 11 \
    --no_replacement_policy
