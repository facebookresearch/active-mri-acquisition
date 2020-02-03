#!/bin/bash

EXTRA_OPTIONS=--no_evaluator

CHECKPOINTS_DIR=/checkpoint/${USER}/active_acq/all_reconstructors_raw_padding_fix

MAKE_DIR=${1:-0}
if [[ ${MAKE_DIR} -eq 1 ]]
    then
    SRC_DIR=/private/home/lep/code/versions/Active_Acquisition/train_reconstructors_$(date +%Y%m%d_%H.%M.%S)
    mkdir -p ${SRC_DIR}
    cp -r /private/home/lep/code/Active_Acquisition/* ${SRC_DIR}
else
    SRC_DIR=/private/home/lep/code/versions/Active_Acquisition/use_correct_version
fi
echo ${SRC_DIR}
cd ${SRC_DIR}

# for mask in basic_rnl symmetric_basic_rnl low_to_high_rnl; do
for mask in symmetric_basic_rnl; do
    export HDF5_USE_FILE_LOCKING=FALSE

    python train_submitit.py --dataroot KNEE \
    --mask_type ${mask} \
    --name ${mask}_DICOM_2 --image_width 128 \
    --checkpoints_dir ${CHECKPOINTS_DIR} \
    --use_submitit \
    --submitit_logs_dir ${CHECKPOINTS_DIR}/submitit_logs \
    --batchSize 40 --mask_embed_dim 0 \
    --gpu_ids 0,1,2,3,4,5,6,7 --print_freq 50 --lr 0.001 --grad_ctx --max_epochs 250 \
    --print_freq 200 --use_mse_as_disc_energy --lambda_gan 0.1 \
    --number_of_cascade_blocks 4 --number_of_reconstructor_filters 128 \
    --n_downsampling 3 --number_of_layers_residual_bottleneck 5 \
    --dropout_probability 0.2 \
    --save_latest_freq 2000 ${EXTRA_OPTIONS} &
done
