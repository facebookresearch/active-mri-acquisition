#!/bin/bash

CHECKPOINTS_BASE=/checkpoint/lep/active_acq
MODELS_DIR=${CHECKPOINTS_BASE}/all_reconstructors
MODEL_TYPE=symmetric_basic_rnl

HOW_MANY_IMAGES=15000

SRC_DIR=/private/home/lep/code/versions/Active_Acquisition/generate_masks_evaluator_$(date +%Y%m%d_%H.%M.%S)

mkdir -p ${SRC_DIR}
cp -r /private/home/lep/code/Active_Acquisition/* ${SRC_DIR}

cd ${SRC_DIR}

JOBSCRIPTS_DIR=/private/home/lep/jobscripts/active_acq
LOGS_DIR=/private/home/lep/logs/active_acq

mkdir -p ${JOBSCRIPTS_DIR}
mkdir -p ${LOGS_DIR}/stdout
mkdir -p ${LOGS_DIR}/stderr

queue=scavenge

# Need to adjust this for train or valid. For train use 1104930, for valid 166000
SPLIT=valid
for initial_index in `seq 0 ${HOW_MANY_IMAGES} 166000`; do
    job_name=dataset_active_acq_${initial_index}

    # This creates a slurm script to call training
    SLURM=${JOBSCRIPTS_DIR}/run.${job_name}.slrm
    echo "#!/bin/bash" > ${SLURM}
    echo "#SBATCH --job-name=$job_name" >> ${SLURM}
    echo "#SBATCH --output=${LOGS_DIR}/stdout/${job_name}.%j" >> ${SLURM}
    echo "#SBATCH --error=${LOGS_DIR}/stderr/${job_name}.%j" >> ${SLURM}
    echo "#SBATCH --partition=$queue" >> ${SLURM}
    echo "#SBATCH --gres=gpu:volta:1" >> ${SLURM}
    echo "#SBATCH --cpus-per-task=1" >> ${SLURM}
    echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
    echo "#SBATCH --mem=24000" >> ${SLURM}
    echo "#SBATCH --time=4320" >> ${SLURM}
    echo "#SBATCH --nodes=1" >> ${SLURM}

    echo "cd ${SRC_DIR}" >> ${SLURM}

    echo srun python generate_masks_evaluator.py \
        --reconstructor_dir ${MODELS_DIR}/${MODEL_TYPE} \
        --evaluator_dir ${MODELS_DIR}/${MODEL_TYPE}/evaluator \
        --dataset_dir ${CHECKPOINTS_BASE}/dataset/round0/${MODEL_TYPE} \
        --initial_index ${initial_index} \
        --test_set ${SPLIT} \
        --how_many_images ${HOW_MANY_IMAGES} >> ${SLURM}

    sbatch ${SLURM}

done
