export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/vae_session'


# name=imagenet_resnet_9blocks_cvae_residual
# python train.py --dataroot 'ImageNet' \
#                 --name $name \
#                 --model ft_cvaenn \
#                 --which_model_netG resnet_9blocks_residual \
#                 --loadSize 144 \
#                 --fineSize 128 \
#                 --norm instance \
#                 --checkpoints_dir '/private/home/zizhao/work/checkpoint_fmri' \
#                 --batchSize 64 \
#                 --niter_decay 100 \
#                 --niter 100 \
#                 --input_nc 2 \
#                 --output_nc 2 \
#                 --no_dropout \
#                 --print_freq 50

name=imagenet_resnet_9blocks_filmvae
mkdir -p -v $checkpoints_dir/$name
cp $checkpoints_dir'/imagenet_resnet_9blocks_attention_residual/200_net_G.pth' $checkpoints_dir/$name/'0_net_G.pth'
python train.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_vaenn \
                --which_model_netG resnet_9blocks_attention_residual_vae \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 64 \
                --niter_decay 100 \
                --niter 100 \
                --input_nc 2 \
                --output_nc 2 \
                --no_dropout \
                --print_freq 50 \
                --train_G \
                --nz 128 \
                --lambda_KL 0.0001