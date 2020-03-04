#!/bin/bash

DATASET=${1:-dicom}
CHECKPOINTS_DIR=${2:-checkpoint}

# Some useful extra options:
#     "--no-evaluator": Trains the reconstruction alone.
#     "--only-evaluator --weights_checkpoint DIR": Trains the evaluator alone, loading
#         the reconstructor weights from DIR.
EXTRA_OPTIONS=${3:-""}


if [[ "${DATASET}" = "dicom" ]]
then
    DATAROOT=KNEE
    MASK=symmetric_basic_rnl
    IW=128
    BS=1
elif [[ "${DATASET}" = "raw" ]]
then
    DATAROOT=KNEE_RAW
    MASK=basic_rnl
    IW=368
    BS=40
else
    echo "Usage ./train_reconstructor [dicom|raw] [checkpoint_dir...] [\"extra_options...\"]"
    exit
fi


cd ..
export HDF5_USE_FILE_LOCKING=FALSE

python trainer.py --dataroot KNEE \
    --mask_type ${MASK} \
    --name ${MASK}_${DATASET} \
    --image_width ${IW} \
    --checkpoints_dir ${CHECKPOINTS_DIR} \
    --batchSize ${BS} \
    --mask_embed_dim 0 \
    --gpu_ids 0,1 \
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
