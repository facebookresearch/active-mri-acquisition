# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_session_v2'

name=knee_pasclassauxgan_uncertainty_w123
python train.py --dataroot 'KNEE' \
                --name $name \
                --model ft_recurgan \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 48 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --print_freq 50 \
                --lambda_gan 1 \
                --use_fixed_weight '1,2,3' \
                --no_kspacemap_embed \
                --which_model_netD 'n_layers_channel_aux'