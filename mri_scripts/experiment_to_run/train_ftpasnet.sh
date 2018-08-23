# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_session_exp'

name=knee_pasnetplus_nouncertainty_w113
python train.py --dataroot 'KNEE' \
                --name $name \
                --model ft_recurnnv2 \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 48 \
                --no_uncertainty \
                --where_loss_weight 'full' \
                --use_fixed_weight '1,1,3' \
                --which_model_netG 'pasnetplus' \
                --gpu_ids $CUDA_VISIBLE_DEVICES 