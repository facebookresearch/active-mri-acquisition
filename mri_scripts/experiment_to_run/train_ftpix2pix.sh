# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_session_exp'

# name=knee_baseline_pixl2pix
# python train.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model pix2pix \
#                 --which_model_netG unet_128 \
#                 --loadSize 144 \
#                 --fineSize 128 \
#                 --norm instance \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 64 \
#                 --niter_decay 50 \
#                 --niter 50 \
#                 --input_nc 2 \
#                 --output_nc 1 \
#                 --print_freq 100 \
#                 --lambda_L1 100 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES \
#                 --dynamic_mask_type 'random' \
#                 --no_dropout \
#                 --verbose
                
# Also train a one with data consistency
# name=knee_baseline_pixl2pix_residual
# python train.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model pix2pix \
#                 --which_model_netG unet_128_residual \
#                 --loadSize 144 \
#                 --fineSize 128 \
#                 --norm instance \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 64 \
#                 --niter_decay 50 \
#                 --niter 50 \
#                 --input_nc 2 \
#                 --output_nc 1 \
#                 --print_freq 100 \
#                 --lambda_L1 100 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES \
#                 --dynamic_mask_type 'random' \
#                 --no_dropout \
#                 --verbose \
#                 --debug

name=knee_resnetzz_baseline_pixl2pix
python train.py --dataroot 'KNEE' \
                --name $name \
                --model pix2pix \
                --which_model_netG 'resnet_residual' \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 64 \
                --niter_decay 50 \
                --niter 50 \
                --input_nc 2 \
                --output_nc 1 \
                --print_freq 100 \
                --lambda_L1 100 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --dynamic_mask_type 'random' \
                --no_dropout \
                --verbose 