# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_session_v2'

# name=knee_energypasnetplus_w1110full_nouc_1gan
# python train.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_pasgan_ablation \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 48 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES \
#                 --print_freq 50 \
#                 --lambda_gan 1 \
#                 --mask_cond \
#                 --where_loss_weight 'full' \
#                 --use_fixed_weight '1,1,10' \
#                 --lr 0.0006 \
#                 --use_mse_as_disc_energy \
#                 --no_uncertanity \
#                 --which_model_netG 'pasnetplus' \
#                 --grad_ctx


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

# name=knee_energypasnetplus_w111logvar_0.1gan_nogradctx_pxlm_groupD
# python train.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_pasgan_ablation \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 48 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES \
#                 --print_freq 50 \
#                 --lambda_gan 0.1 \
#                 --mask_cond \
#                 --where_loss_weight 'logvar' \
#                 --use_fixed_weight '1,1,1' \
#                 --lr 0.0006 \
#                 --use_mse_as_disc_energy \
#                 --which_model_netG 'pasnetplus' \
#                 --which_model_netD 'n_layers_channel_group' \
#                 --ndf 256 \
#                 --n_layers_D 5 \
#                 --pixelwise_loss_merge \
#                 --grad_ctx 

# name=knee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm_nouncatmiddle
# python train.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_pasgan_ablation \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 48 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES \
#                 --print_freq 50 \
#                 --lambda_gan 0.1 \
#                 --mask_cond \
#                 --where_loss_weight 'logvar' \
#                 --use_fixed_weight '1,1,1' \
#                 --lr 0.0006 \
#                 --use_mse_as_disc_energy \
#                 --which_model_netG 'pasnetplus' \
#                 --pixelwise_loss_merge \
#                 --grad_ctx \
#                 --no_uncertanity_at_middle 

# name=knee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm_claaux
# python train.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_pasgan_ablation \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 48 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES \
#                 --print_freq 50 \
#                 --lambda_gan 0.1 \
#                 --mask_cond \
#                 --where_loss_weight 'logvar' \
#                 --use_fixed_weight '1,1,1' \
#                 --lr 0.0006 \
#                 --use_mse_as_disc_energy \
#                 --which_model_netG 'pasnetplus' \
#                 --which_model_netD 'n_layers_channel_aux' \
#                 --pixelwise_loss_merge \
#                 --grad_ctx 

# name=knee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm_classifer
# python train.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_pasgan_ablation \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 48 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES \
#                 --print_freq 50 \
#                 --lambda_gan 0.1 \
#                 --mask_cond \
#                 --where_loss_weight 'logvar' \
#                 --use_fixed_weight '1,1,1' \
#                 --lr 0.0006 \
#                 --use_mse_as_disc_energy \
#                 --which_model_netG 'pasnetplus' \
#                 --which_model_netD 'n_layers_channel_cls_aux' \
#                 --pixelwise_loss_merge \
#                 --grad_ctx \
#                 --debug

name=knee_energyresnetplus_w111logvar_0.03gan
python train.py --dataroot 'KNEE' \
                --name $name \
                --model ft_pasgan_ablation \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 48 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --print_freq 50 \
                --lambda_gan 0.03 \
                --mask_cond \
                --where_loss_weight 'logvar' \
                --use_fixed_weight '1,1,1' \
                --lr 0.0006 \
                --use_mse_as_disc_energy \
                --which_model_netG 'resnetplus' \
                --pixelwise_loss_merge \
                --grad_ctx 