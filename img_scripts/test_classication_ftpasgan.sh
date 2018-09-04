# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_session_exp'

name=imagenet_energyresnetplus_w111logvar_0.1gan_0.1ratio
python test_kspace_recom_classification.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_pasgan_ablation \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 80 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --which_model_netG 'pasnetplus_nomaskcond' \
