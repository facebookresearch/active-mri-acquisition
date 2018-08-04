# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_session_v2'

name=knee_gausian_masking_resnet
python train.py --dataroot 'KNEE' \
                --name $name \
                --model ft_gauscnn \
                --which_model_netG resnet_9blocks_zz \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 64 \
                --niter_decay 100 \
                --niter 100 \
                --input_nc 2 \
                --output_nc 3 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --no_dropout \
                --print_freq 50 \
                --dynamic_mask_type 'random' \
                --debug

