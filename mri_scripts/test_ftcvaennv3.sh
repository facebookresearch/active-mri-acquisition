# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES
set -ex


## randome mask
name=knee_ResCVAEV3_randommask
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_session'

python sampling.py --dataroot 'KNEE' \
                --name $name \
                --model ft_caennv3 \
                --which_model_netG stage_resnet_9blocks_residual_condmask \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 64 \
                --input_nc 2 \
                --output_nc 2 \
                --no_dropout \
                --cvae_attention 'mask' \
                --num_blocks 2 \
                --depth 2 \
                --n_samples 100 \
                --how_many 1000 \
                --ctx_as_residual \
                --gpu_ids $CUDA_VISIBLE_DEVICES 