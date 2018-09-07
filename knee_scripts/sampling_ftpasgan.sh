# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_zz_models'

name=knee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm
python sampling.py --dataroot 'KNEE' \
                --name $name \
                --model ft_pasgan \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 128 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --which_model_netG 'pasnetplus' \
                --how_many 256 