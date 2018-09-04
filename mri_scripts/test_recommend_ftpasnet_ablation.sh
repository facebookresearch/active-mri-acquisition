# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_session_v2'

## general one

# name=knee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm # current best
# python test_kspace_recom.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_pasgan \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 128 \
#                 --which_model_netG 'pasnetplus' \
#                 --how_many 256 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES 


# name=knee_energypasnetplus_w111logvar_0.01gan_gradctx_pxlm # current best
# python test_kspace_recom.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_pasgan \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 128 \
#                 --which_model_netG 'pasnetplus' \
#                 --how_many 256 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES 


# name=knee_energypasnetplus_w111logvar_0.05gan_gradctx_pxlm # current best
# python test_kspace_recom.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_pasgan \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 128 \
#                 --which_model_netG 'pasnetplus' \
#                 --how_many 256 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES 

# name=knee_energypasnetplus_w111logvar_0.2gan_gradctx_pxlm # current best
# python test_kspace_recom.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_pasgan \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 128 \
#                 --which_model_netG 'pasnetplus' \
#                 --how_many 256 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES 

# name=knee_energyresnetplus_w111logvar_0.03gan # current best
# python test_kspace_recom.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_pasgan \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 128 \
#                 --which_model_netG 'resnetplus' \
#                 --how_many 256 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES 


# name=rawknee_energypasnetplus_w111logvar_0.01gan_gradctx_pxlm_gamma20_v2
# name=rawknee_energypasnetplus_w111logvar_0.01gan_gradctx_pxlm_gamma50_v2
# # name=rawknee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm_gamma20
# python test_kspace_recom.py --dataroot 'KNEE_RAW' \
#                 --name $name \
#                 --model ft_pasganraw \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 8 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES \
#                 --how_many 256 
              
name=rawknee_energypasnetplus_w111logvar_0.01gan_gradctx_pxlm_gamma50_v2
python test_kspace_recom.py --dataroot 'KNEE_RAW' \
                --name $name \
                --model ft_pasganraw \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 8 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --how_many 256 