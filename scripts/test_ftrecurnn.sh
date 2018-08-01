# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/cnn_session'

name=imagenet_stage_resnet_9blocks_residual_softeval_randommask
python test.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_recurnn \
                --which_model_netG stage_resnet_9blocks_residual \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 48 \
                --input_nc 2 \
                --output_nc 2 \
                --how_many 20000 \
                --kspace_keep_ratio 0.25 \
                --no_dropout \

# python test.py --dataroot 'ImageNet' \
#                 --name $name \
#                 --model ft_recurnn \
#                 --which_model_netG stage_resnet_9blocks_residual \
#                 --loadSize 144 \
#                 --fineSize 128 \
#                 --norm instance \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 48 \
#                 --input_nc 2 \
#                 --output_nc 2 \
#                 --how_many 20000 \
#                 --kspace_keep_ratio 0.20 \
#                 --no_dropout \


# python test.py --dataroot 'ImageNet' \
#                 --name $name \
#                 --model ft_recurnn \
#                 --which_model_netG stage_resnet_9blocks_residual \
#                 --loadSize 144 \
#                 --fineSize 128 \
#                 --norm instance \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 48 \
#                 --input_nc 2 \
#                 --output_nc 2 \
#                 --how_many 20000 \
#                 --kspace_keep_ratio 0.25 \
#                 --no_dropout \


# python test.py --dataroot 'ImageNet' \
#                 --name $name \
#                 --model ft_recurnn \
#                 --which_model_netG stage_resnet_9blocks_residual \
#                 --loadSize 144 \
#                 --fineSize 128 \
#                 --norm instance \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 48 \
#                 --input_nc 2 \
#                 --output_nc 2 \
#                 --how_many 20000 \
#                 --kspace_keep_ratio 0.3 \
#                 --no_dropout \


# python test.py --dataroot 'ImageNet' \
#                 --name $name \
#                 --model ft_recurnn \
#                 --which_model_netG stage_resnet_9blocks_residual \
#                 --loadSize 144 \
#                 --fineSize 128 \
#                 --norm instance \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 48 \
#                 --input_nc 2 \
#                 --output_nc 2 \
#                 --how_many 20000 \
#                 --kspace_keep_ratio 0.375 \
#                 --no_dropout 