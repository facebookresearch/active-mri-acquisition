# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_session_v2'

name=knee_pasgan_uncertainty_nolsgan_w123_noksemb
python sampling.py --dataroot 'KNEE' \
                --name $name \
                --model ft_recurgan \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 12 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --no_kspacemap_embed \
                --how_many 100 \
                --use_allgen_for_disc
                