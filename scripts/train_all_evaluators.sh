#!/bin/bash

EXTRA_OPTIONS=--only_evaluator

CHECKPOINTS_DIR=/checkpoint/${USER}/active_acq/all_reconstructors_raw_padding_fix

SRC_DIR=/private/home/${USER}/code/versions/Active_Acquisition/train_evaluators_$(date +%Y%m%d_%H.%M.%S)

echo ${SRC_DIR}

mkdir -p ${SRC_DIR}
cp -r /private/home/${USER}/code/Active_Acquisition/* ${SRC_DIR}

cd ${SRC_DIR}

for mask in symmetric_basic_rnl; do
    export HDF5_USE_FILE_LOCKING=FALSE
    python train_submitit.py --dataroot KNEE --image_width 128 \
    --mask_type ${mask} \
    --name ${mask}_DICOM_2_gamma_50 --add_mask_eval --mask_embed_dim 0 \
    --checkpoints_dir ${CHECKPOINTS_DIR} \
    --use_submitit --gamma 50 \
    --submitit_logs_dir ${CHECKPOINTS_DIR}/submitit_logs \
    --batchSize 40 \
    --gpu_ids 0,1,2,3,4,5,6,7 --print_freq 50 --lr 0.0006 --grad_ctx --max_epochs 200 \
    --print_freq 200 --lambda_gan 0.1 \
    --use_mse_as_disc_energy \
    --number_of_cascade_blocks 4 --number_of_reconstructor_filters 128 \
    --n_downsampling 3 --number_of_layers_residual_bottleneck 5 \
    --dropout_probability 0.2 \
    --save_latest_freq 2000 \
    --weights_checkpoint ${CHECKPOINTS_DIR}/${mask}_DICOM_2/best_checkpoint.pth \
    ${EXTRA_OPTIONS} &
done
