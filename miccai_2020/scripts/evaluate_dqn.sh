#!/bin/bash

Help()
{
      echo "Usage:"
      echo "    ./evaluate_dqn -h                                                                                        Display this help message."
      echo "    ./evaluate_dqn dataset_dir data_type mode reconstructor_path checkpoint_dir dqn_weights_dir [metric]     Evaluate DQN model. See below for description of arguments."
      echo "        dataset_dir: Directory storing fastMRI data."
      echo "        data_type: Type of MRI data stored in dataset_dir [raw | dicom]."
      echo "        mode: One of"
      echo "            ss:         Uses subject-specific DQN (image+mask as observation). Acquisition up to 12X acceleration."
      echo "            ss_extreme: Uses subject-specific DQN (image+mask as observation). Acquisition up 100X acceleration."
      echo "            ds:         Uses dataset specific DQN (only time step as observation). Standard acceleration."
      echo "            ds_extreme: Uses dataset specific DQN (only time step as observation). Extreme acceleration."
      echo "        reconstructor_path: Path to the saved reconstructor model."
      echo "        checkpoint_dir: Path to store the results."
      echo "        dqn_weights_dir: Path to the saved DQN model to evaluate. Make sure it was trained in the same mode that it is going to be tested on."
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

DATASET_DIR=${1}
DATA_TYPE=${2}
MODE=${3}
RECONSTR_PATH=${4}
CHECKPOINT_DIR=${5}
DQN_WEIGHTS_DIR=${6}
METRIC=${7:-mse}


# Adjust default options for the desired dataset
if [[ "${DATA_TYPE}" = "dicom" ]]
then
    DATAROOT=KNEE
    INIT_LINES=5
elif [[ "${DATA_TYPE}" = "raw" ]]
then
    DATAROOT=KNEE_RAW
    INIT_LINES=15
else
    echo "Data type not understood."
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
    echo "DDQN mode not understood."
    Help
    exit 1
fi

# Adjust initial number of lines if extreme acceleration is selected
if [[ "${MODE}" = *"_extreme" ]]
then
    INIT_LINES=1
fi

cd ..
export HDF5_USE_FILE_LOCKING=FALSE

python main_miccai20.py \
    --dataset_dir ${DATASET_DIR} \
    --dataroot ${DATAROOT} \
    --reconstructor_path ${RECONSTR_PATH} \
    --checkpoints_dir ${CHECKPOINT_DIR} \
    --dqn_weights_dir ${DQN_WEIGHTS_DIR} \
    --test_set val \
    --num_test_images 1000 \
    --freq_save_test_stats 50 \
    --dqn_test_episode_freq 0 \
    --initial_num_lines_per_side ${INIT_LINES} \
    --obs_type ${OBS_TYPE} \
    --obs_to_numpy \
    --budget 1000 \
    --policy dqn \
    --dqn_model_type ${MODEL_TYPE} \
    --add_mask_eval \
    --dqn_only_test \
    --replay_buffer_size 0