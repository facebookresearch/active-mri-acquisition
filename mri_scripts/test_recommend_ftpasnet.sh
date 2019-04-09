# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/sumanab/checkpoint'

## general one
# name=knee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm # current best
# name=knee_energypasnetplus_w111logvar_0.01gan_gradctx_pxlm # also fine
# name=knee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm_randomfull # similar to the best
# name=knee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm_full
# name=knee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm_2runfixpxlm # just okey
name=dicom_run_1
python test_kspace_recom.py --dataroot 'KNEE' \
                --name $name \
                --model ft_pasgan \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 128 \
                --which_model_netG 'pasnetplus' \
                --gpu_ids $CUDA_VISIBLE_DEVICES 

## two head gan
# name=knee_energypasnetplusclaaux_w113logvar_0.5gan_nogradctx_pxlm
# python test_kspace_recom.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_pasgan_ablation \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 128 \
#                 --which_model_netG 'pasnetplus' \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES \
#                 --which_model_netD 'n_layers_channel_aux' 

# name=knee_energypasnetplus_w111logvar_0.1gan_nogradctx_pxlm_claaux
# python test_kspace_recom.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_pasgan_ablation \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 128 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES \
#                 --which_model_netG 'pasnetplus' \
#                 --which_model_netD 'n_layers_channel_aux' \

# ## group convolution
# name=knee_energypasnetplus_w111logvar_0.1gan_nogradctx_pxlm_groupD
# python test_kspace_recom.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_pasgan_ablation \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 128 \
#                 --which_model_netG 'pasnetplus' \
#                 --which_model_netD 'n_layers_channel_group' \
#                 --ndf 256 \
#                 --n_layers_D 5 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES 

## ablation study: without using energy function
# name=knee_pasgan_uncertainty_w113_logvar_maskmetacond_0.1gan
# python test_kspace_recom.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_pasgan \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 128 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES \
#                 --which_model_netG 'pasnet' \
#                 --how_many 256 \
#                 --n_samples 100

# checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_session_exp'
# ## ablation study: without using adversarial learning 
# name=knee_energynet_pasnetplus_w111logvar_gradctx_pxlm_ablation
# python test_kspace_recom.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_pasnet_ablation \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 128 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES \
#                 --mask_cond \
#                 --where_loss_weight 'logvar' \
#                 --use_fixed_weight '1,1,1' \
#                 --which_model_netG 'pasnetplus' \
