# export CUDA_VISIBLE_DEVICES=$device
# Debug output
set -ex
checkpoint='/private/home/zizhao/work/checkpoint_fmri/cnn_session'
name=imagenet_resnet_9blocks_attention_residual_fullimagenet_v2
python test.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_attcnn \
                --which_model_netG resnet_9blocks_attention_residual \
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
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                | tee $checkpoint/$name/'eval_log.txt'

# python test.py --dataroot 'ImageNet' \
#                 --name $name \
#                 --model ft_attcnn \
#                 --which_model_netG resnet_9blocks_attention_residual \
#                 --loadSize 144 \
#                 --fineSize 128 \
#                 --norm instance \
#                 --checkpoints_dir $checkpoint \
#                 --batchSize 64 \
#                 --input_nc 2 \
#                 --output_nc 2 \
#                 --how_many -1 \
#                 --no_dropout \
#                 --kspace_keep_ratio 0.2 \
#                 | tee -a $checkpoint/$name/'eval_log.txt'

# python test.py --dataroot 'ImageNet' \
#                 --name $name \
#                 --model ft_attcnn \
#                 --which_model_netG resnet_9blocks_attention_residual \
#                 --loadSize 144 \
#                 --fineSize 128 \
#                 --norm instance \
#                 --checkpoints_dir $checkpoint \
#                 --batchSize 64 \
#                 --input_nc 2 \
#                 --output_nc 2 \
#                 --how_many -1 \
#                 --no_dropout \
#                 --kspace_keep_ratio 0.25 \
#                 | tee -a $checkpoint/$name/'eval_log.txt'

# python test.py --dataroot 'ImageNet' \
#                 --name $name \
#                 --model ft_attcnn \
#                 --which_model_netG resnet_9blocks_attention_residual \
#                 --loadSize 144 \
#                 --fineSize 128 \
#                 --norm instance \
#                 --checkpoints_dir $checkpoint \
#                 --batchSize 64 \
#                 --input_nc 2 \
#                 --output_nc 2 \
#                 --how_many -1 \
#                 --no_dropout \
#                 --kspace_keep_ratio 0.3 \
#                 | tee -a $checkpoint/$name/'eval_log.txt'


# python test.py --dataroot 'ImageNet' \
#                 --name $name \
#                 --model ft_attcnn \
#                 --which_model_netG resnet_9blocks_attention_residual \
#                 --loadSize 144 \
#                 --fineSize 128 \
#                 --norm instance \
#                 --checkpoints_dir $checkpoint \
#                 --batchSize 64 \
#                 --input_nc 2 \
#                 --output_nc 2 \
#                 --how_many -1 \
#                 --no_dropout \
#                 --kspace_keep_ratio 0.375 \
#                 | tee -a $checkpoint/$name/'eval_log.txt'