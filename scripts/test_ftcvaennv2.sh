# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES
set -ex

name=imagenet_ResCVAEV2_ctxasres_stdv-6.1618_randommask
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/vae_session'

python test.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_caennv2 \
                --which_model_netG resnet_9blocks_attention_residual \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 16 \
                --input_nc 2 \
                --output_nc 2 \
                --no_dropout \
                --cvae_attention 'mask' \
                --num_blocks 2 \
                --depth 2 \
                --ctx_as_residual \
                --how_many 5000 \
                --n_samples 100 \
                --kspace_keep_ratio 0.125 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --force_use_posterior


python test.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_caennv2 \
                --which_model_netG resnet_9blocks_attention_residual \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 16 \
                --input_nc 2 \
                --output_nc 2 \
                --no_dropout \
                --cvae_attention 'mask' \
                --num_blocks 2 \
                --depth 2 \
                --ctx_as_residual \
                --how_many 5000 \
                --n_samples 100 \
                --kspace_keep_ratio 0.2 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --force_use_posterior

python test.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_caennv2 \
                --which_model_netG resnet_9blocks_attention_residual \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 16 \
                --input_nc 2 \
                --output_nc 2 \
                --no_dropout \
                --cvae_attention 'mask' \
                --num_blocks 2 \
                --depth 2 \
                --ctx_as_residual \
                --how_many 5000 \
                --n_samples 100 \
                --kspace_keep_ratio 0.25 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --force_use_posterior

python test.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_caennv2 \
                --which_model_netG resnet_9blocks_attention_residual \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 16 \
                --input_nc 2 \
                --output_nc 2 \
                --no_dropout \
                --cvae_attention 'mask' \
                --num_blocks 2 \
                --depth 2 \
                --ctx_as_residual \
                --how_many 5000 \
                --n_samples 100 \
                --kspace_keep_ratio 0.3 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --force_use_posterior

python test.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_caennv2 \
                --which_model_netG resnet_9blocks_attention_residual \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 16 \
                --input_nc 2 \
                --output_nc 2 \
                --no_dropout \
                --cvae_attention 'mask' \
                --num_blocks 2 \
                --depth 2 \
                --ctx_as_residual \
                --how_many 5000 \
                --n_samples 100 \
                --kspace_keep_ratio 0.375 \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                --force_use_posterior