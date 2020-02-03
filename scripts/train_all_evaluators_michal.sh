#!/bin/bash

EXTRA_OPTIONS=--only_evaluator

CHECKPOINTS_DIR=/checkpoint/${USER}/active_acq/all_reconstructors_raw_padding_fix

SRC_DIR=/private/home/${USER}/code/versions/Active_Acquisition/train_evaluators_$(date +%Y%m%d_%H.%M.%S)

echo $SRC_DIR

mkdir -p ${SRC_DIR}
cp -r /private/home/${USER}/code/Active_Acquisition/* ${SRC_DIR}

cd ${SRC_DIR}

for mask in basic_rnl; do
    export HDF5_USE_FILE_LOCKING=FALSE
    python train_submitit.py --dataroot KNEE_RAW --image_width 368 \
    --mask_type ${mask} \
    --name ${mask}_data_aug_no_mask_emb_beta=10_post_refactor_gamma_10000 --number_of_evaluator_convolution_layers 4 \
    --checkpoints_dir ${CHECKPOINTS_DIR} --mask_embed_dim 0 --add_mask_eval \
    --use_submitit --gamma 10000 \
    --submitit_logs_dir ${CHECKPOINTS_DIR}/submitit_logs \
    --batchSize 2 \
    --gpu_ids 0,1,2,3,4,5,6,7 --print_freq 50 --lr 0.0006 --grad_ctx --max_epochs 400 \
    --print_freq 200 --use_mse_as_disc_energy --lambda_gan 0.1 \
    --save_latest_freq 2000 \
    --weights_checkpoint ${CHECKPOINTS_DIR}/${mask}_data_aug_no_mask_emb/best_checkpoint.pth \
    ${EXTRA_OPTIONS} &
done
