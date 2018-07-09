# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex

name=imagenet_resnet_9blocks_attention_residual_full_imagenet_dropout
python train.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_attcnn \
                --which_model_netG resnet_9blocks_attention_residual \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir '/private/home/zizhao/work/checkpoint_fmri' \
                --batchSize 96 \
                --niter_decay 50 \
                --niter 50 \
                --input_nc 2 \
                --output_nc 2 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --eval_full_valid 
                # --no_dropout

# name=imagenet_resnet_9blocks_pixelattention_residual
# python train.py --dataroot 'ImageNet' \
#                 --name $name \
#                 --model ft_attcnn \
#                 --which_model_netG resnet_9blocks_pixelattention_residual \
#                 --loadSize 144 \
#                 --fineSize 128 \
#                 --norm instance \
#                 --checkpoints_dir '/private/home/zizhao/work/checkpoint_fmri' \
#                 --batchSize 64 \
#                 --niter_decay 100 \
#                 --niter 100 \
#                 --input_nc 2 \
#                 --output_nc 2 \
#                 --no_dropout
