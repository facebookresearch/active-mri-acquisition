#!/bin/bash

queue=dev

JOBSCRIPTS_DIR=/private/home/lep/jobscripts/active_acq
LOGS_DIR=/private/home/lep/logs/active_acq

mkdir -p ${JOBSCRIPTS_DIR}
mkdir -p ${LOGS_DIR}/stdout
mkdir -p ${LOGS_DIR}/stderr

CHECKPOINTS_BASE=/checkpoint/lep/active_acq
MODELS_DIR=${CHECKPOINTS_BASE}/all_reconstructors_post_eval_tag
MODEL_TYPE=symmetric_basic_rnl

SRC_DIR=/private/home/lep/code/versions/Active_Acquisition/train_dqn_$(date +%Y%m%d_%H.%M.%S)

mkdir -p ${SRC_DIR}
cp -r /private/home/lep/code/Active_Acquisition/* ${SRC_DIR}

cd ${SRC_DIR}

# The lines below create a slurm script to call training
job_name=train_dqn_active_acq

SLURM=${JOBSCRIPTS_DIR}/run.${job_name}.slrm
echo "${SLURM}"

REWARD_METRIC=ssim
NUM_TRAIN_IMG=1000
USE_SCORE_AS_REWARD='--use_score_as_reward'    # Use empty string for false

echo "#!/bin/bash" > ${SLURM}
echo "#SBATCH --job-name=$job_name" >> ${SLURM}
echo "#SBATCH --output=${LOGS_DIR}/stdout/${job_name}.%j" >> ${SLURM}
echo "#SBATCH --error=${LOGS_DIR}/stderr/${job_name}.%j" >> ${SLURM}
echo "#SBATCH --partition=${queue}" >> ${SLURM}
echo "#SBATCH --gres=gpu:volta:1" >> ${SLURM}
echo "#SBATCH --cpus-per-task=2" >> ${SLURM}
echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
echo "#SBATCH --mem=256000" >> ${SLURM}
echo "#SBATCH --time=4320" >> ${SLURM}
echo "#SBATCH --comment=\"Long job that requires cumbersome preemption. CVPR 11/15\"" >> ${SLURM}
echo "#SBATCH --nodes=1" >> ${SLURM}

echo "cd ${SRC_DIR}" >> ${SLURM}

echo srun python acquire_rl.py --dataroot KNEE \
    --reconstructor_dir ${MODELS_DIR}/${MODEL_TYPE} \
    --checkpoints_dir ${MODELS_DIR}/${MODEL_TYPE}/ddqn_tr${NUM_TRAIN_IMG}_val100/${REWARD_METRIC}/ \
    --test_set val \
    --num_train_images ${NUM_TRAIN_IMG} \
    --num_test_images 100 \
    --budget 35 \
    --seed 0 \
    --freq_save_test_stats 100000 \
    --policy dqn \
    --dqn_model_type evaluator \
    --num_train_steps 5000000 \
    --dqn_test_episode_freq 500 \
    --dqn_eval_train_set_episode_freq 3000 \
    --initial_num_lines_per_side 5 \
    --obs_type image_space \
    --obs_to_numpy \
    --gamma 0.5 \
    --dqn_model_type evaluator \
    --dqn_burn_in 1000 \
    --rl_batch_size 16 \
    --reward_metric ${REWARD_METRIC} \
    ${USE_SCORE_AS_REWARD} \
    --replay_buffer_size 200000 >> ${SLURM}

sbatch ${SLURM}
