#!/bin/bash

SRC_DIR=/private/home/lep/code/versions/Active_Acquisition/miccai_exp_post_parallel_eval

cd ${SRC_DIR}

for metric in "mse" "nmse" "ssim" "psnr"; do

    python train_dqn_submitit.py \
    --dataroot KNEE_RAW \
    --partition dev \
    --reconstructor_dir /checkpoint/mdrozdzal/active_acq/all_reconstructors_raw_padding_fix/basic_rnl_RAW_new_data_no_norm_2 \
    --checkpoints_dir /checkpoint/${USER}/active_acq/micca_experiments/raw_data/dqn_normal \
    --job_name dqn_raw_${metric} \
    --gpu_ids 0 \
    --policy dqn \
    --dqn_model_type evaluator \
    --budget 10 \
    --num_test_images 200 \
    --num_train_steps 5000000 \
    --freq_save_test_stats 1000 \
    --initial_num_lines_per_side 15 \
    --test_set val \
    --seed 0 \
    --obs_type image_space \
    --obs_to_numpy \
    --replay_buffer_size 20000 \
    --epsilon_decay 1000000 \
    --dqn_learning_rate 0.0001 \
    --target_net_update_freq 5000 \
    --reward_metric ${metric} \
    --gamma 0.5 \
    --dqn_burn_in 1000 \
    --rl_batch_size 2 \
    --mask_type basic_rnl \
    --test_num_cols_cutoff 100 \
    --add_mask_eval

done


