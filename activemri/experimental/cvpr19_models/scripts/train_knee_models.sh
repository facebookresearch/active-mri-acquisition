# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash

Help()
{
      echo "Usage:"
      echo "    ./train_knee_models -h                                                    Display this help message."
      echo "    ./train_knee_models dataset_dir data_type path [extra_options_...]        Train reconstructor/evaluator model."
      echo "        datset_dir: Directory storing the fastMRI dataset."
      echo "        data_type: Data type of the dataset to use [raw | dicom]."
      echo "        path: location to store model checkpoint and logs."
      echo "        extra_options: Choose at most one from"
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

DATASET_DIR=${1}
DATA_TYPE=${2}
CHECKPOINTS_DIR=${3}
EXTRA_OPTIONS=${4:-""}


if [[ "${DATA_TYPE}" = "dicom" ]]
then
    DATAROOT=KNEE
    MASK=symmetric_basic_rnl
    IW=128
    BS=40
    FILTERS=128
    LAYERS_BOTTLENECK=5
    LR=0.001
    GAMMA=100
elif [[ "${DATA_TYPE}" = "raw" ]]
then
    DATAROOT=KNEE_RAW
    MASK=basic_rnl
    IW=368
    BS=1
    FILTERS=256
    LAYERS_BOTTLENECK=3
    LR=0.0001
    GAMMA=3000
else
    echo "Data type not understood."
    Help
    exit 1
fi

if [[ "${EXTRA_OPTIONS}" == *"only_evaluator"* ]]
then
    LR=0.0006
fi


cd ..
export HDF5_USE_FILE_LOCKING=FALSE

python trainer.py \
    --dataset_dir ${DATASET_DIR} \
    --dataroot ${DATAROOT} \
    --mask_type ${MASK} \
    --name ${MASK}_${DATA_TYPE} \
    --image_width ${IW} \
    --rnl_params 2,4,1,5 \
    --checkpoints_dir ${CHECKPOINTS_DIR} \
    --batchSize ${BS} \
    --gpu_ids 0 \
    --max_epochs 250 \
    --use_mse_as_disc_energy \
    --number_of_cascade_blocks 4 \
    --number_of_reconstructor_filters ${FILTERS} \
    --n_downsampling 3 \
    --number_of_layers_residual_bottleneck ${LAYERS_BOTTLENECK} \
    --dropout_probability 0.2 \
    --mask_embed_dim 0 \
    --lr ${LR} \
    --grad_ctx \
    --number_of_evaluator_convolution_layers 4 \
    --gamma ${GAMMA} \
    --lambda_gan 0.1 \
    --add_mask_eval \
    "${EXTRA_OPTIONS}"
