#!/bin/bash

JOBSCRIPTS_DIR=/private/home/${USER}/jobscripts/active_acq
LOGS_DIR=/private/home/${USER}/logs/active_acq

mkdir -p ${JOBSCRIPTS_DIR}
mkdir -p ${LOGS_DIR}/stdout
mkdir -p ${LOGS_DIR}/stderr

SRC_DIR=/private/home/${USER}/code/versions/Active_Acquisition/train_evaluators_$(date +%Y%m%d_%H.%M.%S)

echo ${SRC_DIR}

mkdir -p ${SRC_DIR}
cp -r /private/home/${USER}/code/Active_Acquisition/* ${SRC_DIR}

queue=dev

job_name=evaluate_dqn_activeacq_dicom

# This creates a slurm script to call training
SLURM=${JOBSCRIPTS_DIR}/run.${job_name}.slrm
echo ${SLURM}

echo "#!/bin/bash" > ${SLURM}
echo "#SBATCH --job-name=$job_name" >> ${SLURM}
echo "#SBATCH --output=${LOGS_DIR}/stdout/${job_name}.%j" >> ${SLURM}
echo "#SBATCH --error=${LOGS_DIR}/stderr/${job_name}.%j" >> ${SLURM}
echo "#SBATCH --partition=$queue" >> ${SLURM}
echo "#SBATCH --gres=gpu:volta:8" >> ${SLURM}
echo "#SBATCH --constraint=volta32gb" >> ${SLURM}
echo "#SBATCH --cpus-per-task=16" >> ${SLURM}
echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
echo "#SBATCH --mem=64000" >> ${SLURM}
echo "#SBATCH --time=4320" >> ${SLURM}
echo "#SBATCH --comment=\"\"" >> ${SLURM}
echo "#SBATCH --nodes=1" >> ${SLURM}

echo "cd ${SRC_DIR}" >> ${SLURM}

echo CUDA_VISIBLE_DEVICES=0 srun python acquire_rl.py --dataroot KNEE \
    --reconstructor_dir /checkpoint/mdrozdzal/active_acq/all_reconstructors_raw_padding_fix/symmetric_basic_rnl_DICOM_2 \
    --evaluator_dir dummy \
    --checkpoints_dir /checkpoint/mdrozdzal/active_acq/all_reconstructors_raw_padding_fix/symmetric_basic_rnl_DICOM_2/dqn/image_space.tupd5000.bs16.edecay1000000.0.gamma0.5.lr0.0001.repbuf200000norepl1.nimgtr1000000.metricmse.usescoasrew0_bu35_seed0_neptest100/eval \
    --dqn_weights_dir /checkpoint/mdrozdzal/active_acq/all_reconstructors_raw_padding_fix/symmetric_basic_rnl_DICOM_2/dqn/image_space.tupd5000.bs16.edecay1000000.0.gamma0.5.lr0.0001.repbuf200000norepl1.nimgtr1000000.metricmse.usescoasrew0_bu35_seed0_neptest100 \
    --test_set val \
    --num_test_images 1000 \
    --freq_save_test_stats 50 \
    --dqn_test_episode_freq 0 \
    --initial_num_lines_per_side 5 \
    --obs_type image_space \
    --obs_to_numpy \
    --budget 1000 \
    --policy dqn \
    --mask_type symmetric_basic_rnl \
    --dqn_model_type evaluator \
    --add_mask_eval \
    --dqn_only_test \
    --replay_buffer_size 0 >> ${SLURM}

sbatch ${SLURM}
