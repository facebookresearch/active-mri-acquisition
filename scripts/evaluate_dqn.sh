#!/bin/bash

JOBSCRIPTS_DIR=/private/home/lep/jobscripts/active_acq
LOGS_DIR=/private/home/lep/logs/active_acq

mkdir -p ${JOBSCRIPTS_DIR}
mkdir -p ${LOGS_DIR}/stdout
mkdir -p ${LOGS_DIR}/stderr

CHECKPOINTS_BASE=/checkpoint/lep/active_acq
MODELS_DIR=${CHECKPOINTS_BASE}/all_reconstructors_post_eval_tag
MODEL_TYPE=symmetric_basic_rnl

#SRC_DIR=/private/home/lep/code/versions/Active_Acquisition/test_dqn_$(date +%Y%m%d_%H.%M.%S)
SRC_DIR=/private/home/lep/code/versions/Active_Acquisition/eval_many_dqns
#rm -rf ${SRC_DIR}
#mkdir -p ${SRC_DIR}
#cp -r /private/home/lep/code/Active_Acquisition/* ${SRC_DIR}
#echo ${SRC_DIR}

queue=dev

INIT_LINES=5
BASELINES_SUFFIX=init.num.lines.${INIT_LINES}

DQN_STR=image_space.tupd5000.bs16.edecay500000.gamma0.0.lr0.0001.repbuf300000norepl1.nimgtr10000000.metricssim.usescoasrew0_bu35_seed0_neptest100
SUBDIR=long_dqns

job_name=evaluate_dqn_activeacq__${MODEL_TYPE}__${DQN_STR}

# This creates a slurm script to call training
SLURM=${JOBSCRIPTS_DIR}/run.${job_name}.slrm
echo ${SLURM}

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
echo "#SBATCH --comment=\"CVPR deadline 11/15\"" >> ${SLURM}
echo "#SBATCH --nodes=1" >> ${SLURM}

echo "cd ${SRC_DIR}" >> ${SLURM}

echo srun python acquire_rl.py --dataroot KNEE \
    --reconstructor_dir ${MODELS_DIR}/${MODEL_TYPE}\
    --evaluator_dir ${MODELS_DIR}/${MODEL_TYPE}/evaluator \
    --checkpoints_dir ${MODELS_DIR}/${MODEL_TYPE}/eval_${SUBDIR}/${DQN_STR} \
    --dqn_weights_dir ${MODELS_DIR}/${MODEL_TYPE}/${SUBDIR}/${DQN_STR} \
    --test_set test \
    --num_test_images 1000 \
    --freq_save_test_stats 50 \
    --dqn_test_episode_freq 0 \
    --initial_num_lines_per_side 5 \
    --obs_type image_space \
    --obs_to_numpy \
    --budget 1000 \
    --policy dqn \
    --dqn_model_type evaluator \
    --dqn_only_test \
    --replay_buffer_size 0 >> ${SLURM}

sbatch ${SLURM}
