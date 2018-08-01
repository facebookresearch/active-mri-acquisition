# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/cnn_session'

name=imagenet_stage_resnet_9blocks_residual_softeval_randommask2
python train.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_recurnn \
                --which_model_netG stage_resnet_9blocks_residual \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 48 \
                --niter_decay 100 \
                --niter 100 \
                --input_nc 2 \
                --output_nc 2 \
                --print_freq 100 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --dynamic_mask_type 'random' \
                --no_dropout \
                --debug 

# name=imagenet_sideout_resnet_9blocks_residual_softeval
# python train.py --dataroot 'ImageNet' \
#                 --name $name \
#                 --model ft_recurnn \
#                 --which_model_netG so_resnet_9blocks_residual \
#                 --loadSize 144 \
#                 --fineSize 128 \
#                 --norm instance \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 48 \
#                 --niter_decay 100 \
#                 --niter 100 \
#                 --input_nc 2 \
#                 --output_nc 2 \
#                 --print_freq 100 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES \
#                 --no_dropout 