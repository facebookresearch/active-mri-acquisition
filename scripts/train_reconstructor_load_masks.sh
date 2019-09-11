#!/bin/bash

MODEL_TYPE=symmetric_basic_rnl
EXTRA_OPTIONS=--no_evaluator
ROUND=0

BASE_DIR=/checkpoint/${USER}/active_acq
CHECKPOINTS_DIR=${BASE_DIR}/all_reconstructors

SRC_DIR=/private/home/lep/code/versions/Active_Acquisition/train_reconstructors_$(date +%Y%m%d_%H.%M.%S)

echo $SRC_DIR

mkdir -p ${SRC_DIR}
cp -r /private/home/lep/code/Active_Acquisition/* ${SRC_DIR}

cd ${SRC_DIR}

python train_submitit.py \
    --dataroot KNEE_PRECOMPUTED_MASKS \
    --mask_type basic \
    --checkpoints_dir ${CHECKPOINTS_DIR} \
    --name round$((ROUND + 1))_${MODEL_TYPE} \
    --use_submitit \
    --submitit_logs_dir ${CHECKPOINTS_DIR}/submitit_logs \
    --batchSize 40 \
    --gpu_ids 0,1,2,3,4,5,6,7 --print_freq 50 --lr 0.0006 --grad_ctx --max_epochs 50 \
    --print_freq 200 --use_mse_as_disc_energy --lambda_gan 0.1 \
    --weights_checkpoint ${CHECKPOINTS_DIR}/${MODEL_TYPE}/best_checkpoint.pth \
    --masks_dir ${BASE_DIR}/dataset/round${ROUND}/${MODEL_TYPE} \
    --save_latest_freq 2000 ${EXTRA_OPTIONS} &
