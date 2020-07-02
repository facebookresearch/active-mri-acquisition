#!/bin/bash

Help()
{
      echo "Usage:"
      echo "    ./train_knee_models -h                                      Display this help message."
      echo "    ./train_knee_models dataset path [extra_options_...]        Train reconstructor/evaluator model."
      echo "        dataset: Dataset to use [raw | dicom]."
      echo "        path: location to store model checkpoint and logs."
      echo "        extra_options: Zero or more from"
      echo "            --no_evaluator: Trains the reconstructor alone."
      echo "            --only_evaluator --weights_checkpoint <path>: Trains evaluator alone, loading reconstructor weights from path."
}

while getopts ":h" opt; do
  case ${opt} in
    h )
     Help
     exit 0
     ;;
  esac
done

DATASET=${1}
CHECKPOINTS_DIR=${2}
EXTRA_OPTIONS=${3:-""}


if [[ "${DATASET}" = "dicom" ]]
then
    DATAROOT=KNEE
    MASK=symmetric_basic_rnl
    IW=128
    BS=40
elif [[ "${DATASET}" = "raw" ]]
then
    DATAROOT=KNEE_RAW
    MASK=basic_rnl
    IW=368
    BS=1
else
    Help
    exit 1
fi


cd ..
export HDF5_USE_FILE_LOCKING=FALSE

python trainer.py --dataroot ${DATAROOT} \
    --mask_type ${MASK} \
    --name ${MASK}_${DATASET} \
    --image_width ${IW} \
    --checkpoints_dir ${CHECKPOINTS_DIR} \
    --batchSize ${BS} \
    --mask_embed_dim 0 \
    --gpu_ids 0 \
    --print_freq 50 \
    --lr 0.001 \
    --grad_ctx \
    --max_epochs 250 \
    --print_freq 200 \
    --use_mse_as_disc_energy \
    --lambda_gan 0.1 \
    --number_of_cascade_blocks 4 \
    --number_of_reconstructor_filters 128 \
    --n_downsampling 3 \
    --number_of_layers_residual_bottleneck 5 \
    --dropout_probability 0.2 \
    --save_latest_freq 2000 \
    ${EXTRA_OPTIONS}
