# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES
set -ex


## randome mask
name=knee_ResCVAEV3_randommask
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_session'
mkdir -p -v $checkpoints_dir/$name
cp $checkpoints_dir'/knee_pasnetcm_randommask/10_net_G.pth' $checkpoints_dir/$name/'0_net_G.pth'

python train.py --dataroot 'KNEE' \
                --name $name \
                --model ft_caennv3 \
                --which_model_netG stage_resnet_9blocks_residual_condmask \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 32 \
                --niter_decay 50 \
                --niter 50 \
                --input_nc 2 \
                --output_nc 2 \
                --no_dropout \
                --preload_G \
                --lr 0.0005 \
                --cvae_attention 'mask' \
                --beta 1 \
                --num_blocks 2 \
                --depth 2 \
                --print_freq 50 \
                --ctx_as_residual \
                --dynamic_mask_type 'random' \
                --gpu_ids $CUDA_VISIBLE_DEVICES 