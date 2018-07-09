# # jure unet
# set -ex
# name=imagenet_unet
# python test.py --dataroot 'ImageNet' \
#                 --name $name \
#                 --model ft_cnn \
#                 --which_model_netG jure_unet \
#                 --loadSize 144 \
#                 --fineSize 128 \
#                 --checkpoints_dir '/private/home/zizhao/work/checkpoint_fmri' \
#                 --batchSize 128 \
#                 --input_nc 2 \
#                 --output_nc 1 \
#                 --how_many -1 \
#                 | tee -a $path/$name/'val_log.txt' 

# set -ex
name=imagenet_resnet_9blocks_residual_1c_3down_v3
python test.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_cnn \
                --which_model_netG resnet_9blocks_residual \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir '/private/home/zizhao/work/checkpoint_fmri' \
                --batchSize 64 \
                --how_many 256 \
                --input_nc 2 \
                --output_nc 1 \
                --how_many -1 \
                --no_dropout 