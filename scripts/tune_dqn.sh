#!/bin/bash

MODELS_DIR=/checkpoint/lep/active_acq/all_reconstructors_no_sign_leakage_michal
MODEL_TYPE=symmetric_choice_rnl

SRC_DIR=/private/home/lep/code/versions/Active_Acquisition/tune_dqn_$(date +%Y%m%d_%H.%M.%S)

mkdir -p ${SRC_DIR}
cp -r /private/home/lep/code/Active_Acquisition/* ${SRC_DIR}

cd ${SRC_DIR}

python tune_dqn.py --dataroot KNEE \
    --reconstructor_dir ${MODELS_DIR}/${MODEL_TYPE}\
    --evaluator_dir ${MODELS_DIR}/${MODEL_TYPE}/evaluator \
    --checkpoints_dir /checkpoint/lep/active_acq/debug/tune_dqn_test2/ \
    --test_set val \
    --num_test_images 100 \
    --freq_save_test_stats 100000 \
    --dqn_test_episode_freq 20 \
    --sequential_images \
    --initial_num_lines 5 \
    --obs_type two_streams \
    --dqn_resume \
    --rl_env_train_no_seed \
    --budget 65 \
    --use_score_as_reward \
    --no_replacement_policy
