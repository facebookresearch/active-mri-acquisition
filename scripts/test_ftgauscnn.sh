# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/cnn_session'

name=imagenet_gausian_masking_resnet2
python sampling.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_gauscnn \
                --which_model_netG resnet_9blocks_zz \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 16 \
                --n_samples 100 \
                --input_nc 2 \
                --output_nc 3 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --no_dropout \
                --which_epoch 60 \

