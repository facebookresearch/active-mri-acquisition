#!/bin/bash

JOBSCRIPTS_DIR=/private/home/lep/jobscripts/active_acq
LOGS_DIR=/private/home/lep/logs/active_acq

mkdir -p ${JOBSCRIPTS_DIR}
mkdir -p ${LOGS_DIR}/stdout
mkdir -p ${LOGS_DIR}/stderr

CHECKPOINTS_BASE=/checkpoint/lep/active_acq
MODELS_DIR=${CHECKPOINTS_BASE}/all_reconstructors_refactor_rl_env
MODEL_TYPE=symmetric_basic_rnl

SRC_DIR=/private/home/lep/code/versions/Active_Acquisition/test_dqn_$(date +%Y%m%d_%H.%M.%S)

queue=dev

INIT_LINES=10
BASELINES_SUFFIX=init.num.lines.${INIT_LINES}

WEIGHTS_DIR=ddqn_tr1000_val100/image_space.tupd5000.bs16.edecay500000.gamma0.5.norepl1.nimgtr1000._bu20_seed0_neptest100

job_name=evaluate_dqn_activeacq

# This creates a slurm script to call training
SLURM=${JOBSCRIPTS_DIR}/run.${job_name}.slrm
echo "#!/bin/bash" > ${SLURM}
echo "#SBATCH --job-name=$job_name" >> ${SLURM}
echo "#SBATCH --output=${LOGS_DIR}/stdout/${job_name}.%j" >> ${SLURM}
echo "#SBATCH --error=${LOGS_DIR}/stderr/${job_name}.%j" >> ${SLURM}
echo "#SBATCH --partition=$queue" >> ${SLURM}
echo "#SBATCH --gres=gpu:volta:1" >> ${SLURM}
echo "#SBATCH --cpus-per-task=2" >> ${SLURM}
echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
echo "#SBATCH --mem=64000" >> ${SLURM}
echo "#SBATCH --time=4320" >> ${SLURM}
# echo "#SBATCH --comment=\"NeurIPS deadline 05/23\"" >> ${SLURM}
echo "#SBATCH --nodes=1" >> ${SLURM}

echo "cd /private/home/lep/code/Active_Acquisition" >> ${SLURM}


echo srun python acquire_rl.py --dataroot KNEE \
    --reconstructor_dir ${MODELS_DIR}/${MODEL_TYPE}\
    --evaluator_dir ${MODELS_DIR}/${MODEL_TYPE}/evaluator \
    --checkpoints_dir ${MODELS_DIR}/${MODEL_TYPE}/dqn \
    --dqn_weights_dir ${MODELS_DIR}/${MODEL_TYPE}/${WEIGHTS_DIR} \
    --test_set test \
    --num_test_images 1000 \
    --freq_save_test_stats 50 \
    --dqn_test_episode_freq 0 \
    --sequential_images \
    --initial_num_lines_per_side 5 \
    --obs_type image_space \
    --obs_to_numpy \
    --budget 1000 \
    --policy dqn \
    --dqn_model_type evaluator \
    --dqn_only_test \
    --replay_buffer_size 0 >> ${SLURM}

sbatch ${SLURM}
