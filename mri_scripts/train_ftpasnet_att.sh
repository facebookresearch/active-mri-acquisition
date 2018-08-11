# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_session_v2'

name=knee_pasnet_att_uncertainty_w111_noatthidden
python train.py --dataroot 'KNEE' \
                --name $name \
                --model ft_recurnnv2 \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 48 \
                --which_model_netG 'pasnet_att' \
                --gpu_ids $CUDA_VISIBLE_DEVICES                 