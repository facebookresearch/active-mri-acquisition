#!/bin/bash

Help()
{
      echo "Usage:"
      echo "    ./evaluate_baseline -h                                                                               Display this help message."
      echo "    ./evaluate_baseline dataset_dir data_type reconstructor_dir checkpoint_path policy [evaluator_dir]   Evaluate baseline. See below for description of arguments."
      echo "        datset_dir: Directory storing the fastMRI dataset."
      echo "        data_type: Data type of the dataset to use [raw | dicom]."
      echo "        reconstructor_path: Path to the saved reconstructor model."
      echo "        checkpoint_dir: Path to store the results."
      echo "        policy: The baseline policy to use. One of [random | lowfirst| random_low_bias | evaluator_net | one_step_greedy]."
      echo "        evaluator_path: (Optional) Path to evaluator model, when policy=evaluator_net."
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
RECONSTR_PATH=${3}
CHECKPOINT_DIR=${4}
POLICY=${5:-random}
EVALUATOR_PATH=${6:-""}

EVALUATOR_ARGS=""
if [[ "${EVALUATOR_PATH}" != "" ]]
then
    EVALUATOR_ARGS="--evaluator_path ${EVALUATOR_PATH} --add_mask_eval"
fi

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

cd ..
export HDF5_USE_FILE_LOCKING=FALSE

# budget and test_num_cols_cutoff are set to large values so that maximum values are used.
python main_miccai20.py \
    --dataset_dir ${DATASET_DIR} \
    --dataroot ${DATAROOT} \
    --mask_type basic_rnl \
    --reconstructor_path ${RECONSTR_PATH} ${EVALUATOR_ARGS} \
    --checkpoints_dir ${CHECKPOINT_DIR} \
    --seed 0 \
    --gpu_ids 0 \
    --policy ${POLICY} \
    --obs_type image_space \
    --add_mask_eval \
    --initial_num_lines_per_side ${INIT_LINES} \
    --budget 1000 \
    --num_test_images 1000 \
    --freq_save_test_stats 50 \
    --test_set val \
    --test_num_cols_cutoff 500 \
