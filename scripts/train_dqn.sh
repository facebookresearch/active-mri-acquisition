#!/bin/bash

Help()
{
      echo "Usage:"
      echo "    ./train_dqn -h                                                        Display this help message."
      echo "    ./train_dqn dataset mode reconstructor_dir checkpoint_dir [metric]    Train DQN model. See below for description of arguments."
      echo "        dataset: Dataset to use [raw | dicom]."
      echo "        mode: One of"
      echo "            ss:         Uses subject-specific DQN (image+mask as observation). Acquisition up to 12X acceleration."
      echo "            ss_extreme: Uses subject-specific DQN (image+mask as observation). Acquisition up 100X acceleration."
      echo "            ds:         Uses dataset specific DQN (only time step as observation). Standard acceleration."
      echo "            ds_extreme: Uses dataset specific DQN (only time step as observation). Extreme acceleration."
      echo "        reconstructor_dir: Path to the saved reconstructor model."
      echo "        checkpoint_dir: Path to store model and logs of DQN training."
      echo "        metric: (Optional) Reward metric to use from [mse|nmse|ssim|psnr]. Default: mse."
}

while getopts ":h" opt; do
  case ${opt} in
    h )
     Help
     exit 0
     ;;
  esac
done

set -x

DATASET=${1}
MODE=${2}
RECONSTR_DIR=${3}
CHECKPOINT_DIR=${4}
METRIC=${5:-mse}


# Adjust default options for the desired dataset
if [[ "${DATASET}" = "dicom" ]]
then
    DATAROOT=KNEE
    RL_BS=32
    RB_SIZE=200000
    COLS_CUTOFF=30
    MASK_TYPE=symmetric_basic_rnl
    INIT_LINES=5
    BUDGET=20
elif [[ "${DATASET}" = "raw" ]]
then
    DATAROOT=KNEE_RAW
    RL_BS=2
    RB_SIZE=20000
    COLS_CUTOFF=100
    MASK_TYPE=basic_rnl
    INIT_LINES=15
    BUDGET=70
else
    Help
    exit 1
fi

# Adjust options for dataset-specific vs. subject-specific
if [[ "${MODE}" = "ds"* ]]
then
    MODEL_TYPE=simple_mlp
    OBS_TYPE=only_mask
elif [[ "${MODE}" = "ss"* ]]
then
    MODEL_TYPE=evaluator
    OBS_TYPE=image_space
else
    Help
    exit 1
fi

# Adjust budget if extreme acceleration is selected
if [[ "${MODE}" = *"_extreme" ]]
then
    INIT_LINES=1
    if [[ "${DATASET}" = "dicom" ]]
    then
        BUDGET=28
    else
        BUDGET=98
    fi
fi

cd ..
export HDF5_USE_FILE_LOCKING=FALSE

python acquire_rl.py \
    --dataroot ${DATAROOT} \
    --reconstructor_dir ${RECONSTR_DIR} \
    --checkpoints_dir ${CHECKPOINT_DIR} \
    --gpu_ids 0 \
    --policy dqn \
    --dqn_model_type ${MODEL_TYPE} \
    --budget ${BUDGET} \
    --num_test_images 200 \
    --num_train_steps 5000000 \
    --freq_save_test_stats 1000 \
    --initial_num_lines_per_side ${INIT_LINES} \
    --test_set val \
    --seed 0 \
    --obs_type ${OBS_TYPE} \
    --obs_to_numpy \
    --replay_buffer_size ${RB_SIZE} \
    --epsilon_decay 1000000 \
    --dqn_learning_rate 0.0001 \
    --target_net_update_freq 5000 \
    --reward_metric ${METRIC} \
    --gamma 0.5 \
    --dqn_burn_in 1000 \
    --rl_batch_size ${RL_BS} \
    --mask_type basic_rnl \
    --test_num_cols_cutoff ${COLS_CUTOFF} \
    --add_mask_eval \
    --train_with_fixed_initial_mask
