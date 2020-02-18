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
    SRC_DIR=/private/home/lep/code/versions/Active_Acquisition/dicom_greedy_and_open_loop
fi
echo ${SRC_DIR}

queue=dev

INIT_LINES=15
BASELINES_SUFFIX=init.num.lines.${INIT_LINES}_2

MAX_NUM_EVALS=${3:-None}
TEST_BUDGET=${4:-None}

indices=(0 400 800 1200 1600 1851)

for i in $(seq 1 5); do
    start=${indices[$(($i-1))]}
    diff=$((${indices[$i]} - $start))

    obs_type=image_space
    job_name=greedy_raw_${start}_${TEST_BUDGET}_${MAX_NUM_EVALS}

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
    echo "#SBATCH --constraint=volta32gb" >> ${SLURM}
    echo "#SBATCH --time=4320" >> ${SLURM}
    # echo "#SBATCH --comment=\"NeurIPS deadline 05/23\"" >> ${SLURM}
    echo "#SBATCH --nodes=1" >> ${SLURM}

    echo "cd ${SRC_DIR}" >> ${SLURM}

    echo export HDF5_USE_FILE_LOCKING=FALSE >> ${SLURM}

    echo srun python acquire_rl.py --dataroot KNEE_RAW --mask_type basic_rnl \
    --reconstructor_dir /checkpoint/mdrozdzal/active_acq/all_reconstructors_raw_padding_fix/basic_rnl_RAW_new_data \
    --evaluator_dir None \
    --checkpoints_dir /checkpoint/lep/active_acq/all_reconstructors_raw_padding_fix/basic_rnl_basic_RAW_new_data/all_baselines.${BASELINES_SUFFIX}/$2_${TEST_BUDGET}_${MAX_NUM_EVALS}/start_${start} \
    --seed 0 --gpu_ids 0 --policy one_step_greedy \
    --num_train_steps 0 --num_train_images 0 \
    --obs_type ${obs_type} \
    --add_mask_eval \
    --initial_num_lines_per_side ${INIT_LINES} \
    --test_budget ${TEST_BUDGET} \
    --greedy_max_num_actions ${MAX_NUM_EVALS} \
    --reward_metric $2 \
    --num_test_images ${diff} \
    --test_set_shift ${start} \
    --freq_save_test_stats 50 --test_set test >> ${SLURM}

     sbatch ${SLURM}

     sleep 2
done