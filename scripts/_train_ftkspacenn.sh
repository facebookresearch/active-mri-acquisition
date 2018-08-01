export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES
set -ex

name=imagenet_kspace_unet_residual
python train.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_kspace \
                --which_model_netG kspace_unet_residual \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir '/private/home/zizhao/work/checkpoint_fmri' \
                --batchSize 64 \
                --niter_decay 100 \
                --niter 100 \
                --input_nc 256 \
                --output_nc 256 \
                --no_dropout \
                --print_freq 50 \

