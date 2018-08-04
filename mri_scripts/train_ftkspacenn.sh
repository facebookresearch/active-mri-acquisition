# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
name=knee_kspace_vsrnet_randommask_dilated
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_session'
mkdir -p -v $checkpoints_dir/$name
cp $checkpoints_dir'/knee_pasnetcm_randommask/100_net_G.pth' $checkpoints_dir/$name/'0_net_G_basis.pth'


python train.py --dataroot 'KNEE' \
                --name $name \
                --model ft_kspace \
                --which_model_netG stage_resnet_9blocks_residual_condmask \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 64 \
                --niter_decay 20 \
                --niter 10 \
                --input_nc 2 \
                --output_nc 2 \
                --print_freq 10 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --dynamic_mask_type 'random' \
                --no_dropout \
                --use_dilation