# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/sumanab/checkpoint/'

name=dicom_run_1
python sampling.py --dataroot 'KNEE' \
                --name $name \
                --model ft_pasgan \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 128 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --which_model_netG 'pasnetplus' \
                --how_many 256 