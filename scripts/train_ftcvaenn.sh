# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES
set -ex

# name=imagenet_ResCVAE_beta4_ctxasres
# checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/vae_session'
# mkdir -p -v $checkpoints_dir/$name
# cp $checkpoints_dir'/imagenet_resnet_9blocks_attention_residual/200_net_G.pth' $checkpoints_dir/$name/'0_net_G.pth'

# python train.py --dataroot 'ImageNet' \
#                 --name $name \
#                 --model ft_caenn \
#                 --which_model_netG resnet_9blocks_attention_residual \
#                 --loadSize 144 \
#                 --fineSize 128 \
#                 --norm instance \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 32 \
#                 --niter_decay 50 \
#                 --niter 50 \
#                 --input_nc 2 \
#                 --output_nc 2 \
#                 --no_dropout \
#                 --preload_G \
#                 --lr 0.00005 \
#                 --cvae_attention 'mask' \
#                 --beta 4 \
#                 --num_blocks 2 \
#                 --depth 2 \
#                 --print_freq 50 \
#                 --ctx_gen_det_skip False \
#                 --ctx_as_residual \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES 

## retrain from other model
name=imagenet_ResCVAE_beta0.1to4_ctxasres_fixedVI2
which_epoch=50
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/vae_session'
mkdir -p -v $checkpoints_dir/$name
cp $checkpoints_dir'/imagenet_ResCVAE_beta.1_ctxasres/'$which_epoch'_net_G.pth' $checkpoints_dir/$name/$which_epoch'_net_G.pth'
cp $checkpoints_dir'/imagenet_ResCVAE_beta.1_ctxasres/'$which_epoch'_net_CVAE.pth' $checkpoints_dir/$name/$which_epoch'_net_CVAE.pth'
# cp $checkpoints_dir'/imagenet_ResCVAE_beta.1_ctxasres/'$which_epoch'_optim_CVAE.pth' $checkpoints_dir/$name/$which_epoch'_optim_CVAE.pth' 

python train.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_scaenn \
                --which_model_netG resnet_9blocks_attention_residual \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 32 \
                --niter_decay 50 \
                --niter 50 \
                --input_nc 2 \
                --output_nc 2 \
                --no_dropout \
                --preload_G \
                --lr 0.00005 \
                --cvae_attention 'mask' \
                --beta 4 \
                --num_blocks 2 \
                --depth 2 \
                --print_freq 10 \
                --ctx_gen_det_skip False \
                --ctx_as_residual \
                --continue_train \
                --non_strict_state_dict \
                --which_epoch $which_epoch \
                --gpu_ids $CUDA_VISIBLE_DEVICES 

## ablation study
# name=imagenet_resnet_CVAEuasinf_beta2_ctxasres
# checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/vae_session'
# mkdir -p -v $checkpoints_dir/$name
# cp $checkpoints_dir'/imagenet_resnet_9blocks_attention_residual/200_net_G.pth' $checkpoints_dir/$name/'0_net_G.pth'

# python train.py --dataroot 'ImageNet' \
#                 --name $name \
#                 --model ft_cae2nn \
#                 --which_model_netG resnet_9blocks_attention_residual \
#                 --loadSize 144 \
#                 --fineSize 128 \
#                 --norm instance \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 32 \
#                 --niter_decay 50 \
#                 --niter 50 \
#                 --input_nc 2 \
#                 --output_nc 2 \
#                 --no_dropout \
#                 --preload_G \
#                 --lr 0.00005 \
#                 --cvae_attention 'mask' \
#                 --beta 4 \
#                 --num_blocks 2 \
#                 --depth 2 \
#                 --print_freq 50 \
#                 --ctx_gen_det_skip False \
#                 --ctx_as_residual \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES 