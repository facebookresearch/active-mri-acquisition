
# # jure unet
# set -ex
# name=imagenet_unet
# CUDA_VISIBLE_DEVICES=$device python train.py --dataroot 'ImageNet' \
#                 --name $name \
#                 --model ft_cnn \
#                 --which_model_netG jure_unet \
#                 --loadSize 144 \
#                 --fineSize 128 \
#                 --checkpoints_dir '/private/home/zizhao/work/checkpoint_fmri' \
#                 --batchSize 128 \
#                 --niter_decay 50 \
#                 --niter 200


set -ex
name=imagenet_resnet9bzz
CUDA_VISIBLE_DEVICES=$device python train.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_cnn \
                --which_model_netG resnet_9blocks_zz \
                --loadSize 144 \
                --fineSize 128 \
                --checkpoints_dir '/private/home/zizhao/work/checkpoint_fmri' \
                --batchSize 128 \
                --niter_decay 50 \
                --niter 200