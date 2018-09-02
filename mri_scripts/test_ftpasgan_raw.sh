# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_session_v2'

name=rawknee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm
python sampling.py --dataroot 'KNEE_RAW' \
                --name $name \
                --model ft_pasganraw \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 8 \
                --gpu_ids $CUDA_VISIBLE_DEVICES 