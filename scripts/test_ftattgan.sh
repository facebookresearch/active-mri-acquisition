# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/cnn_session'

name=imagenet_masking_gan_gan2c_l12c_6ld_l11k_dlr2m
python test.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_attgan \
                --which_model_netG resnet_9blocks_zz \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 64 \
                --input_nc 2 \
                --output_nc 2 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --no_dropout \
                --how_many -1

