#!/bin/bash

CHECKPOINTS_BASE=/checkpoint/lep/active_acq
MODELS_DIR=${CHECKPOINTS_BASE}/all_reconstructors_refactor_rl_env
MODEL_TYPE=symmetric_basic_rnl

SRC_DIR=/private/home/lep/code/versions/Active_Acquisition/train_dqn_$(date +%Y%m%d_%H.%M.%S)

mkdir -p ${SRC_DIR}
cp -r /private/home/lep/code/Active_Acquisition/* ${SRC_DIR}

cd ${SRC_DIR}

python acquire_rl.py --dataroot KNEE \
    --reconstructor_dir ${MODELS_DIR}/${MODEL_TYPE}\
    --evaluator_dir ${MODELS_DIR}/${MODEL_TYPE}/evaluator \
    --checkpoints_dir ${MODELS_DIR}/dqt_test_1image \
    --test_set train \
    --num_test_images 1 \
    --freq_save_test_stats 100000 \
    --dqn_test_episode_freq 50 \
    --sequential_images \
    --initial_num_lines_per_side 5 \
    --obs_type image_space \
    --obs_to_numpy \
    --budget 20 \
    --dqn_model_type evaluator \
    --replay_buffer_size 50000
