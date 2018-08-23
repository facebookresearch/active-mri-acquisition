# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_session_v2'

# claaux model actually 
name=knee_energypasnetplusclaaux_w113logvar_0.5gan_nogradctx_pxlm
python train.py --dataroot 'KNEE' \
                --name $name \
                --model ft_pasgan_ablation \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 48 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --print_freq 50 \
                --lambda_gan 0.5 \
                --mask_cond \
                --where_loss_weight 'logvar' \
                --use_fixed_weight '1,1,3' \
                --lr 0.0006 \
                --use_mse_as_disc_energy \
                --which_model_netG 'pasnetplus' \
                --grad_ctx \
                --pixelwise_loss_merge \
                --which_model_netD 'n_layers_channel_aux' 