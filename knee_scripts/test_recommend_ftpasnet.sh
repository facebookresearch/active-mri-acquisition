# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_zz_models'

## general one
name=knee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm 
# name=knee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm_full_randomfull # trained on the full set
python test_kspace_recom.py --dataroot 'KNEE' \
                --name $name \
                --model ft_pasgan \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 128 \
                --which_model_netG 'pasnetplus' \
                --gpu_ids $CUDA_VISIBLE_DEVICES 

## for raw data 
# name=rawknee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm_gamma20_v3_randomfull
# python test_kspace_recom.py --dataroot 'KNEE_RAW' \
#                 --name $name \
#                 --model ft_pasganraw \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 8 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES \