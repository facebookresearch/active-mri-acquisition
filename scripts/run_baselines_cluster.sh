#!/bin/bash

JOBSCRIPTS_DIR=/private/home/${USER}/jobscripts/active_acq
LOGS_DIR=/private/home/${USER}/logs/active_acq

mkdir -p ${JOBSCRIPTS_DIR}
mkdir -p ${LOGS_DIR}/stdout
mkdir -p ${LOGS_DIR}/stderr

MAKE_DIR=${1:-0}
if [[ ${MAKE_DIR} -eq 1 ]]
then
    SRC_DIR=/private/home/lep/code/versions/Active_Acquisition/run_baselines_$(date +%Y%m%d_%H.%M.%S)
    mkdir -p ${SRC_DIR}
    cp -r /private/home/lep/code/Active_Acquisition/* ${SRC_DIR}
else
    SRC_DIR=/private/home/lep/code/versions/Active_Acquisition/run_baselines_20200210_13.10.20
fi
echo ${SRC_DIR}

queue=dev

INIT_LINES=5
BASELINES_SUFFIX=init.num.lines.${INIT_LINES}_2
EXP_SUBSTR=ZZ_SCORE

EXTRA_OPTIONS="--eval_with_zz_score"

for NAME in "symmetric_basic_rnl"; do
    RECONSTRUCTOR_DIR=/checkpoint/mdrozdzal/active_acq/all_reconstructors_raw_padding_fix/${NAME}_DICOM_2
    EVALUATOR_DIR=${RECONSTRUCTOR_DIR}_gamma_1000/evaluator
    CHECKPOINT_DIR=/checkpoint/${USER}/active_acq/all_reconstructors_raw_padding_fix/${NAME}_DICOM_2/${EXP_SUBSTR}/all_baselines.${BASELINES_SUFFIX}
    for policy in "random" "lowfirst" "evaluator_net"; do
        obs_type=image_space
        job_name=active_acq_baselines_${EXP_SUBSTR}_${NAME}_${policy}

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
        echo "#SBATCH --nodes=1" >> ${SLURM}

        echo "cd ${SRC_DIR}" >> ${SLURM}

        echo export HDF5_USE_FILE_LOCKING=FALSE >> ${SLURM}

        echo srun python acquire_rl.py --dataroot KNEE --mask_type ${NAME} \
        --reconstructor_dir ${RECONSTRUCTOR_DIR} \
        --evaluator_dir ${EVALUATOR_DIR} \
        --checkpoints_dir ${CHECKPOINT_DIR} \
        --seed 0 --gpu_ids 0 --policy ${policy} \
        --num_train_steps 0 --num_train_images 0 \
        --obs_type ${obs_type} \
        --add_mask_eval \
        --initial_num_lines_per_side ${INIT_LINES} \
        --budget 1000 \
        --num_test_images 1000 \
        --freq_save_test_stats 50 --test_set test \
        ${EXTRA_OPTIONS} >> ${SLURM}

         sbatch ${SLURM}
    done
done
