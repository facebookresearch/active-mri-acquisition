# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_zz_models'

# gamma is the weight of discriminator energy function need to be careful here
name=rawknee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm_gamma20_v3_run2
python train.py --dataroot 'KNEE_RAW' \
                --name $name \
                --model ft_pasganraw \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 4 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --print_freq 50 \
                --lambda_gan 0.1 \
                --mask_cond \
                --where_loss_weight 'logvar' \
                --use_fixed_weight '1,1,1' \
                --lr 0.0002 \
                --niter 20 \
                --niter 30 \
                --gamma 20 \
                --eval_full_valid