# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES
set -ex
## 4GPUS batchSize 24 beta 0.01 work well

name=imagenet_resnet_CVAE_maskatt_gpus_beta.01
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/vae_session'
mkdir -p -v $checkpoints_dir/$name
cp $checkpoints_dir'/imagenet_resnet_9blocks_attention_residual/200_net_G.pth' $checkpoints_dir/$name/'0_net_G.pth'

python train.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_caenn \
                --which_model_netG resnet_9blocks_attention_residual \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 24 \
                --niter_decay 100 \
                --niter 100 \
                --input_nc 2 \
                --output_nc 2 \
                --no_dropout \
                --print_freq 50 \
                --preload_G \
                --which_epoch 0 \
                --lr 0.00005 \
                --cvae_attention 'mask' \
                --print_freq 50 \
                --beta 0.01 \
                --gpu_ids $CUDA_VISIBLE_DEVICES 

                # --train_G \
