# set -ex
# name=imagent_pix2pix
# CUDA_VISIBLE_DEVICES=$device python train.py --dataroot 'ImageNet' \
#                 --name $name \
#                 --model pix2pix \
#                 --which_model_netG resnet_9blocks_zz \
#                 --loadSize 144 \
#                 --fineSize 128 \
#                 --lambda_L1 100 \
#                 --norm instance \
#                 --pool_size 0 \
#                 --checkpoints_dir '/private/home/zizhao/work/checkpoint_fmri' \
#                 --input_nc 1 \
#                 --output_nc 1 \
#                 --batchSize 32 \
#                 --no_cond_gan


set -ex
name=imagent_pix2pix_perceploss
CUDA_VISIBLE_DEVICES=$device python train.py --dataroot 'ImageNet' \
                --name $name \
                --model pix2pix \
                --which_model_netG resnet_9blocks_zz \
                --loadSize 144 \
                --fineSize 128 \
                --lambda_L1 50 \
                --norm instance \
                --pool_size 0 \
                --checkpoints_dir '/private/home/zizhao/work/checkpoint_fmri' \
                --input_nc 1 \
                --output_nc 1 \
                --batchSize 32 \
                --niter_decay 50 \
                --lambda_vgg 50 \
                --continue_train \
                --epoch_count 50 \
