#!/bin/bash

EXTRA_OPTIONS=--no_evaluator

NUM_VOL_TRAIN=1000
NUM_VOL_VAL=$((NUM_VOL_TRAIN / 10))
CHECKPOINTS_DIR=/checkpoint/${USER}/active_acq/smaller_dicom_dataset/num_vol_train_${NUM_VOL_TRAIN}


#SRC_DIR=/private/home/lep/code/versions/Active_Acquisition/train_reconstructors_$(date +%Y%m%d_%H.%M.%S)
SRC_DIR=/private/home/lep/code/versions/Active_Acquisition/train_reconstructors_dicom_smaller

echo ${SRC_DIR}

mkdir -p ${SRC_DIR}
cp -r /private/home/lep/code/Active_Acquisition/* ${SRC_DIR}

cd ${SRC_DIR}

for mask in basic_rnl symmetric_basic_rnl low_to_high_rnl; do
    python train_submitit.py --dataroot KNEE \
    --mask_type ${mask} \
    --name ${mask} \
    --checkpoints_dir ${CHECKPOINTS_DIR} \
    --use_submitit \
    --submitit_logs_dir ${CHECKPOINTS_DIR}/submitit_logs \
    --batchSize 40 \
    --gpu_ids 0,1,2,3,4,5,6,7 --print_freq 50 --lr 0.0006 --grad_ctx --max_epochs 100 \
    --print_freq 200 --use_mse_as_disc_energy --lambda_gan 0.1 \
    --num_volumes_train ${NUM_VOL_TRAIN} \
    --num_volumes_val ${NUM_VOL_VAL} \
    --save_latest_freq 2000 ${EXTRA_OPTIONS} &
done
