export CUDA_VISIBLE_DEVICES=$device

# # jure unet
# set -ex
# name=imagenet_unet
# python test.py --dataroot 'ImageNet' \
#                 --name $name \
#                 --model ft_cnn \
#                 --which_model_netG jure_unet_residual \
#                 --loadSize 144 \
#                 --fineSize 128 \
#                 --checkpoints_dir '/private/home/zizhao/work/checkpoint_fmri' \
#                 --batchSize 128 \
#                 --input_nc 2 \
#                 --output_nc 1 \
#                 --how_many -1 \
#                 | tee -a $path/$name/'val_log.txt' 

# set -ex
export CUDA_VISIBLE_DEVICES=$device
# Debug output
set -ex
checkpoint='/private/home/zizhao/work/checkpoint_fmri/cnn_session'
name=imagenet_unet_residual_2c_randommask
python test.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_cnn \
                --which_model_netG jure_unet_residual \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoint \
                --batchSize 64 \
                --input_nc 2 \
                --output_nc 2 \
                --how_many -1 \
                --no_dropout \
                --kspace_keep_ratio 0.125 \
                | tee -a $checkpoint/$name/'eval_log.txt'

python test.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_cnn \
                --which_model_netG jure_unet_residual \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoint \
                --batchSize 64 \
                --input_nc 2 \
                --output_nc 2 \
                --how_many -1 \
                --no_dropout \
                --kspace_keep_ratio 0.2 \
                | tee -a $checkpoint/$name/'eval_log.txt'

python test.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_cnn \
                --which_model_netG jure_unet_residual \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoint \
                --batchSize 64 \
                --input_nc 2 \
                --output_nc 2 \
                --how_many -1 \
                --no_dropout \
                --kspace_keep_ratio 0.25 \
                | tee -a $checkpoint/$name/'eval_log.txt'

python test.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_cnn \
                --which_model_netG jure_unet_residual \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoint \
                --batchSize 64 \
                --input_nc 2 \
                --output_nc 2 \
                --how_many -1 \
                --no_dropout \
                --kspace_keep_ratio 0.3 \
                | tee -a $checkpoint/$name/'eval_log.txt'


python test.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_cnn \
                --which_model_netG jure_unet_residual \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoint \
                --batchSize 64 \
                --input_nc 2 \
                --output_nc 2 \
                --how_many -1 \
                --no_dropout \
                --kspace_keep_ratio 0.375 \
                | tee -a $checkpoint/$name/'eval_log.txt'