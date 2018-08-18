# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_session_v2'

name=knee_pasgan_uncertainty_w113_logvar_maskmetacond_0.5gan_mseenergy_lr6e-4
python sampling.py --dataroot 'KNEE' \
                --name $name \
                --model ft_recurgan \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 128 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --how_many 256 

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