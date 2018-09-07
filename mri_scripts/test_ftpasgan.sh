# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_session_exp'

# name=knee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm
# name=knee_pasnetplus_w112_0.5gan_gradctx # sharper
name=knee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm_full
python sampling.py --dataroot 'KNEE' \
                --name $name \
                --model ft_pasgan \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 128 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --which_model_netG 'pasnetplus' \
                --how_many 256 \
                --kspace_keep_ratio 0.1
                
# name=knee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm_full_randomfull
# python sampling.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_pasgan \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 128 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES \
#                 --which_model_netG 'pasnetplus' \
#                 --how_many 256 

# name=knee_paslocalgan_w113_1gan
# python sampling.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_recurgan \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 128 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES \
#                 --how_many 256 \
#                 --which_model_netD 'n_layers' \
#                 --n_layers_D 3 \
#                 --which_epoch 70 \
#                 --ndf 64

# name=knee_pasgan_uncertainty_w113_logvar_maskmetacond_0.1gan
# python sampling.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_pasgan \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 128 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES \
#                 --which_model_netG 'pasnet' \
#                 --how_many 256 \
#                 --n_samples 100

# checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_session_v2'
# ## raw data
# name=rawknee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm_gamma20_v3_randomfull
# python sampling.py --dataroot 'KNEE_RAW' \
#                 --name $name \
#                 --model ft_pasganraw \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 8 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES \
#                 --mask_cond \
#                 --how_many 256 \
#                 --kspace_keep_ratio 0.1