# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/cnn_session'

name=imagenet_stage_resnet_9blocks_residual_allksm_noskipadd
python train.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_recurnn \
                --which_model_netG stage_resnet_9blocks_residual \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 48 \
                --niter_decay 100 \
                --niter 100 \
                --input_nc 2 \
                --output_nc 2 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --no_dropout  
