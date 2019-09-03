#!/bin/bash

EXTRA_OPTIONS=--only_evaluator

CHECKPOINTS_DIR=/checkpoint/${USER}/active_acq/all_reconstructors_no_sign_leakage_michal

SRC_DIR=/private/home/lep/code/versions/Active_Acquisition/train_evaluators_$(date +%Y%m%d_%H.%M.%S)

echo $SRC_DIR

mkdir -p ${SRC_DIR}
cp -r /private/home/lep/code/Active_Acquisition/* ${SRC_DIR}

cd ${SRC_DIR}

for mask in fixed_acc fixed_acc_rnl symmetric_choice symmetric_choice_rnl grid symmetric_grid; do
    python train_submitit.py --dataroot KNEE \
    --mask_type ${mask} \
    --name ${mask} \
    --checkpoints_dir ${CHECKPOINTS_DIR} \
    --use_submitit \
    --submitit_logs_dir ${CHECKPOINTS_DIR}/submitit_logs \
    --batchSize 40 \
    --gpu_ids 0,1,2,3,4,5,6,7 --print_freq 50 --lr 0.0006 --grad_ctx --max_epochs 50 \
    --print_freq 200 --use_mse_as_disc_energy --lambda_gan 0.1 \
    --save_latest_freq 2000 \
    --weights_checkpoint ${CHECKPOINTS_DIR}/${mask}/best_checkpoint.pth \
    ${EXTRA_OPTIONS} &
done
