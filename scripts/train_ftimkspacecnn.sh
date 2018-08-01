# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES
set -ex

name=imagenet_imkspace_net_coordconv
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/cnn_session' 

# # jure unet
python train.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_imkspace \
                --which_model_netG resnet_9blocks_fourier_coordconv \
                --loadSize 144 \
                --fineSize 128 \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 54 \
                --input_nc 2 \
                --output_nc 2 \
                --niter_decay 100 \
                --niter 100 \
                --print_freq 100 \
                --gpu_ids $CUDA_VISIBLE_DEVICES 