
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES
set -ex
name=imagenet_resnet_9blocks_residual_uncondgan_l2
python train.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_pix2pix \
                --which_model_netG resnet_9blocks_residual \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir '/private/home/zizhao/work/checkpoint_fmri' \
                --batchSize 64 \
                --niter_decay 100 \
                --niter 200 \
                --input_nc 2 \
                --output_nc 1 \
                --no_dropout \
                --lambda_L1 100 \
                --pool_size 0 \
                --no_cond_gan
