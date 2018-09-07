# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_zz_models'

## basic lr for 1gpu is 0.0002, so 0.0006 if train with 6 gpus (I do not think it need to go larger even with more gpus)
## eval_full_valid: evalute the whole dataset
name=knee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm_run2
python train.py --dataroot 'KNEE' \
                --name $name \
                --model ft_pasgan \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 48 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --print_freq 50 \
                --lambda_gan 0.1 \
                --mask_cond \
                --where_loss_weight 'logvar' \
                --use_fixed_weight '1,1,1' \
                --lr 0.0006 \
                --use_mse_as_disc_energy \
                --which_model_netG 'pasnetplus' \
                --pixelwise_loss_merge \
                --grad_ctx \
                --niter 50 \
                --niter 50 \
                # --eval_full_valid \ 
                # --dynamic_mask_type 'random_full' 


