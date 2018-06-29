export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES
set -ex

name=imagenet_resnet_9blocks_vae_residual
python train.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_vaenn \
                --which_model_netG resnet_9blocks_vae_residual \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir '/private/home/zizhao/work/checkpoint_fmri' \
                --batchSize 64 \
                --niter_decay 100 \
                --niter 100 \
                --input_nc 2 \
                --output_nc 1 \
                --no_dropout \
                --print_freq 100