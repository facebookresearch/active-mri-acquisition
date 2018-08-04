# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_session'

name=knee_pasnetcm_randommask
python test.py --dataroot 'KNEE' \
                --name $name \
                --model ft_recurnn \
                --which_model_netG stage_resnet_9blocks_residual_condmask \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 128 \
                --input_nc 2 \
                --output_nc 2 \
                --how_many 512 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --no_dropout 