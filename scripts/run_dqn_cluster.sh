#!/bin/bash

JOBSCRIPTS_DIR=/private/home/${USER}/jobscripts/active_acq/dqn
LOGS_DIR=/private/home/${USER}/logs/active_acq/dqn

RESULTS_DIR=/checkpoint/${USER}/active_acq/train_with_evaluator_symmetric

mkdir -p ${JOBSCRIPTS_DIR}
mkdir -p ${LOGS_DIR}/stdout
mkdir -p ${LOGS_DIR}/stderr
mkdir -p ${RESULTS_DIR}

queue=dev
budget=20

for gamma in 0.0 0.25 0.5 0.75; do
for num_images in 1 5 10 20; do
    job_name=active_acq_dqn_gamma.${gamma}_num_images.${num_images}

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
    echo "#SBATCH --mem=128000" >> ${SLURM}
    echo "#SBATCH --time=4320" >> ${SLURM}
    # echo "#SBATCH --comment=\"NeurIPS deadline 05/23\"" >> ${SLURM}
    echo "#SBATCH --nodes=1" >> ${SLURM}

    echo "cd /private/home/lep/code/Active_Acquisition" >> ${SLURM}

    echo srun --label python acquire_rl.py --dataroot KNEE --policy dqn --sequential_images \
    --checkpoints_dir /checkpoint/lep/active_acq/train_with_evaluator_symmetric \
    --results_dir ${RESULTS_DIR} \
    --rl_logs_subdir dqn_test_with_train_budget${budget} \
    --no_replacement_policy \
    --batchSize 96 --gpu_ids 0 \
    --num_train_episodes 10000 \
    --epsilon_decay 150000 \
    --target_net_update_freq 1000 \
    --rl_batch_size 32 \
    --gamma ${gamma} \
    --budget ${budget} \
    --num_test_images ${num_images} --num_train_images ${num_images} \
    --freq_save_test_stats ${num_images} \
    --agent_test_episode_freq 100 --replay_buffer_size 100000 >> ${SLURM}

     sbatch ${SLURM}
done
done