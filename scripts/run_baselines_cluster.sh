#!/bin/bash

JOBSCRIPTS_DIR=/private/home/lep/jobscripts/active_acq
LOGS_DIR=/private/home/lep/logs/active_acq

mkdir -p ${JOBSCRIPTS_DIR}
mkdir -p ${LOGS_DIR}/stdout
mkdir -p ${LOGS_DIR}/stderr

queue=learnfair

BASELINES_SUFFIX=init.num.lines.8

for NAME in "fixed_acc" "fixed_acc_rnl" "symmetric_choice" "symmetric_choice_rnl" "grid" "symmetric_grid"; do
    CHECKPOINT_DIR=/checkpoint/lep/active_acq/all_reconstructors/${NAME}
    for policy in "random" "lowfirst" "evaluator_net"; do
    #for policy in "evaluator++"; do
        if [[ ${policy} == "evaluator++" ]]
        then
            obs_type=concatenate_mask
        else
            obs_type=two_streams
        fi
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

        echo "cd /private/home/lep/code/Active_Acquisition" >> ${SLURM}

        echo srun python acquire_rl.py --dataroot KNEE \
        --reconstructor_dir ${CHECKPOINT_DIR} \
        --evaluator_dir ${CHECKPOINT_DIR}/evaluator \
        --checkpoints_dir ${CHECKPOINT_DIR}/all_baselines_${BASELINES_SUFFIX} \
        --seed 0 --batchSize 96 --gpu_ids 0 --policy ${policy} \
        --num_train_steps 0 --num_train_images 0 \
        --evaluator_pp_path ${CHECKPOINT_DIR}/evaluator_pp_15k/bs_256_lr_0.0003_beta1_0.5_beta2_0.999/best_checkpoint.pth \
        --obs_type ${obs_type} \
        --greedymc_num_samples 60 --greedymc_horizon 1 \
        --sequential_images \
        --initial_num_lines 8 \
        --budget 1000 \
        --num_test_images 1000 \
        --freq_save_test_stats 50 \
        --use_reconstructions >> ${SLURM}

         sbatch ${SLURM}
    done
done
