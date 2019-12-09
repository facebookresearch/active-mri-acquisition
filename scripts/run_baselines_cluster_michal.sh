#!/bin/bash

JOBSCRIPTS_DIR=/private/home/${USER}/jobscripts/active_acq
LOGS_DIR=/private/home/${USER}/logs/active_acq

mkdir -p ${JOBSCRIPTS_DIR}
mkdir -p ${LOGS_DIR}/stdout
mkdir -p ${LOGS_DIR}/stderr

queue=learnfair

INIT_LINES=16
BASELINES_SUFFIX=init.num.lines.${INIT_LINES}_301



for NAME in "basic_rnl"; do
    CHECKPOINT_DIR=/checkpoint/${USER}/active_acq/all_reconstructors_raw_padding_fix/${NAME}_data_aug_no_mask_emb
    for policy in "random" "lowfirst" "evaluator_net"; do
        obs_type=image_space
        job_name=active_acq_baselines_${NAME}_${policy}

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

        echo "cd /private/home/mdrozdzal/code/Active_Acquisition" >> ${SLURM}

        echo srun python acquire_rl.py --dataroot KNEE_RAW \
        --reconstructor_dir ${CHECKPOINT_DIR} \
        --evaluator_dir ${CHECKPOINT_DIR}_beta=10_new_loss_4/evaluator \
        --checkpoints_dir ${CHECKPOINT_DIR}_beta=10_new_loss_4/all_baselines.${BASELINES_SUFFIX} \
        --seed 0 --batchSize 10 --gpu_ids 0 --policy ${policy} \
        --num_train_steps 0 --num_train_images 0 \
        --obs_type ${obs_type} \
        --greedymc_num_samples 60 --greedymc_horizon 1 \
        --sequential_images \
        --initial_num_lines_per_side ${INIT_LINES} \
        --budget 1000 \
        --num_test_images 1000 \
        --freq_save_test_stats 50 --test_set val >> ${SLURM}

         sbatch ${SLURM}
    done
done
