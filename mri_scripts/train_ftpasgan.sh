# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_session_v2'

name=knee_energypasnetplus_w111logvar_0.01gan_gradctx_pxlm
python train.py --dataroot 'KNEE' \
                --name $name \
                --model ft_pasgan \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 48 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --print_freq 50 \
                --lambda_gan 0.01 \
                --mask_cond \
                --where_loss_weight 'logvar' \
                --use_fixed_weight '1,1,1' \
                --lr 0.0006 \
                --use_mse_as_disc_energy \
                --which_model_netG 'pasnetplus' \
                --pixelwise_loss_merge \
                --grad_ctx

# name=knee_paslocalgan_w111_10gan
# python train.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_pasgan \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 48 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES \
#                 --print_freq 50 \
#                 --lambda_gan 10 \
#                 --mask_cond \
#                 --where_loss_weight 'logvar' \
#                 --use_fixed_weight '1,1,1' \
#                 --lr 0.0006 \
#                 --which_model_netD 'n_layers' \
#                 --n_layers_D 3 \
#                 --ndf 64 