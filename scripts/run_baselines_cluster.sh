#!/bin/bash

JOBSCRIPTS_DIR=/private/home/lep/jobscripts/active_acq
LOGS_DIR=/private/home/lep/logs/active_acq

mkdir -p ${JOBSCRIPTS_DIR}
mkdir -p ${LOGS_DIR}/stdout
mkdir -p ${LOGS_DIR}/stderr

queue=learnfair

use_reconstruction=1

# for policy in "random" "lowfirst" "greedyfull1_gt" "evaluator_net" "greedyfull1nors_gt" "evaluator++"; do
for policy in "evaluator++"; do
    if [[ ${policy} == "evaluator++" ]]
    then
        rl_obs_type=concatenate_mask
    else
        rl_obs_type=two_streams
    fi
    if [[ ${use_reconstruction} -eq 1 ]]
    then
        policy=${policy}_r
    fi
    job_name=active_acq_baselines_${policy}

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
    --checkpoints_dir /checkpoint/lep/active_acq/full_test_run_py --checkpoint_suffix 38_mse=0.02237745 \
    --seed 0 --batchSize 96 --gpu_ids 0 --policy ${policy} \
    --num_train_episodes 0 --num_train_images 0 \
    --budget 1000 --num_test_images 1000 --freq_save_test_stats 40 --sequential_images \
    --rl_logs_subdir all_baselines_best_mse \
    --evaluator_pp_path evaluator_pp_15k/bs_256_lr_0.0003_beta1_0.5_beta2_0.999/best_checkpoint.pth \
    --rl_obs_type ${rl_obs_type} \
    --greedymc_num_samples 60 --greedymc_horizon 1 >> ${SLURM}

     sbatch ${SLURM}
done
