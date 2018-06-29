export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES
set -ex

# # jure unet
# name=imagenet_unet_residual_2c_l2norm
# python train.py --dataroot 'ImageNet' \
#                 --name $name \
#                 --model ft_cnn \
#                 --which_model_netG jure_unet_residual \
#                 --loadSize 144 \
#                 --fineSize 128 \
#                 --checkpoints_dir '/private/home/zizhao/work/checkpoint_fmri' \
#                 --batchSize 128 \
#                 --input_nc 2 \
#                 --output_nc 2 \
#                 --niter_decay 100 \
#                 --niter 100 \
#                 --l2_weight 

name=imagenet_resnet_9blocks_residual_1c_3down_v3
python train.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_cnn \
                --which_model_netG resnet_9blocks_residual \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir '/private/home/zizhao/work/checkpoint_fmri' \
                --batchSize 64 \
                --niter_decay 100 \
                --niter 100 \
                --input_nc 2 \
                --output_nc 1 \
                --no_dropout


name=imagenet_resnet_9blocks_residual_1c_3down_v3
python train.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_cnn \
                --which_model_netG resnet_9blocks_residual \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir '/private/home/zizhao/work/checkpoint_fmri' \
                --batchSize 64 \
                --niter_decay 100 \
                --niter 100 \
                --input_nc 2 \
                --output_nc 1 \
                --no_dropout