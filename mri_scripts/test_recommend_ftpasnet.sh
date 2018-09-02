# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_session_v2'

## general one
# name=knee_pasgan_uncertainty_w113_logvar_maskmetacond_0.5gan_mseenergy_lr6e-4
# name=knee_energypasnetplus_w113logvar_0.5gan_gradctx_pxlm
name=knee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm # current best
# name=knee_energypasnetplus_w111logvar_0.01gan_gradctx_pxlm # also fine
python test_kspace_recom.py --dataroot 'KNEE' \
                --name $name \
                --model ft_pasgan \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 128 \
                --which_model_netG 'pasnetplus' \
                --how_many 256 \
                --gpu_ids $CUDA_VISIBLE_DEVICES 

## two head gan
# name=knee_energypasnetplusclaaux_w113logvar_0.5gan_nogradctx_pxlm
# python test_kspace_recom.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_pasgan_ablation \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 128 \
#                 --which_model_netG 'pasnetplus' \
#                 --how_many 256 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES \
#                 --which_model_netD 'n_layers_channel_aux' 

# name=knee_energypasnetplus_w111logvar_0.1gan_nogradctx_pxlm_claaux
# python test_kspace_recom.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_pasgan_ablation \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 128 \
#                 --how_many 256 \
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
#                 --how_many 256 \
#                 --which_model_netD 'n_layers_channel_group' \
#                 --ndf 256 \
#                 --n_layers_D 5 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES 

## 
