echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_session_v2'
batchSize=64

# name=knee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm
name=knee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm_2runfixpxlm
python test_moving_ratio.py --dataroot 'KNEE' \
                --name $name \
                --model ft_pasgan_ablation \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 128 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --which_model_netG 'pasnetplus' \

# name=knee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm_nouncatmiddle
# python test_moving_ratio.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_pasgan_ablation \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 128 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES \
#                 --which_model_netG 'pasnetplus' \

# checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/mri_session_exp'

# name=knee_pasnetplus_uncertainty_w111
# python test_moving_ratio.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_recurnnv2 \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 48 \
#                 --which_model_netG 'pasnetplus' \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES 

# name=knee_pasnetplus_nouncertainty_w111
# python test_moving_ratio.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_recurnnv2 \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 48 \
#                 --which_model_netG 'pasnetplus' \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES        

# name=knee_baseline_resnetzz
# python test_moving_ratio.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_attcnn \
#                 --which_model_netG 'resnet_residual' \
#                 --loadSize 144 \
#                 --fineSize 128 \
#                 --norm instance \
#                 --input_nc 2 \
#                 --output_nc 2 \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize $batchSize \
#                 --no_dropout 

# name=knee_baseline_denset103
# python test_moving_ratio.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_attcnn \
#                 --which_model_netG densenet \
#                 --loadSize 144 \
#                 --fineSize 128 \
#                 --norm instance \
#                 --input_nc 2 \
#                 --output_nc 1 \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize $batchSize \
#                 --no_dropout 

# name=knee_baseline_unet
# python test_moving_ratio.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model ft_attcnn \
#                 --which_model_netG jure_unet_residual \
#                 --loadSize 144 \
#                 --fineSize 128 \
#                 --norm instance \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize $batchSize \
#                 --input_nc 2 \
#                 --output_nc 1 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES \
#                 --no_dropout 

# name=knee_resnetzz_baseline_pixl2pix
# python test_moving_ratio.py --dataroot 'KNEE' \
#                 --name $name \
#                 --model pix2pix \
#                 --which_model_netG 'resnet_residual' \
#                 --loadSize 144 \
#                 --fineSize 128 \
#                 --norm instance \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize $batchSize \
#                 --input_nc 2 \
#                 --output_nc 1 \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES \
#                 --no_dropout 