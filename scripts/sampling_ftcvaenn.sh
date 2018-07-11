export CUDA_VISIBLE_DEVICES=$device

name=imagenet_resnet_CVAE_maskatt_gpus_beta.01
checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/vae_session'
set -ex
python sampling.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_caenn \
                --which_model_netG resnet_9blocks_attention_residual \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 24 \
                --input_nc 2 \
                --output_nc 2 \
                --no_dropout \
                --how_many 256 \
                --n_samples 64 \
