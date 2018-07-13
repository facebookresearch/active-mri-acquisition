export CUDA_VISIBLE_DEVICES=$device
set -ex
# name=imagenet_unet_vae
# python test.py --dataroot 'ImageNet' \
#                 --name $name \
#                 --model ft_vaenn \
#                 --which_model_netG jure_unet_vae_residual \
#                 --loadSize 144 \
#                 --fineSize 128 \
#                 --norm instance \
#                 --checkpoints_dir '/private/home/zizhao/work/checkpoint_fmri' \
#                 --input_nc 2 \
#                 --output_nc 1 \
#                 --batchSize 64 \
#                 --how_many 256 \
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/vae_session'

name=imagenet_resnet_9blocks_filmvae
python sampling.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_vaenn \
                --which_model_netG resnet_9blocks_attention_residual_vae \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 20 \
                --input_nc 2 \
                --output_nc 2 \
                --no_dropout \
                --nz 128 \
                --n_samples 64 \
