# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_session_exp'

name=imagenet_energyresnetplus_w111logvar_0.1gan_0.1ratio
python train.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_pasgan_ablation \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 48 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --print_freq 50 \
                --lambda_gan 0.1 \
                --where_loss_weight 'logvar' \
                --use_fixed_weight '1,1,1' \
                --lr 0.0006 \
                --use_mse_as_disc_energy \
                --which_model_netG 'pasnetplus_nomaskcond' \
                --pixelwise_loss_merge \
                --grad_ctx \
                --kspace_keep_ratio 0.1 