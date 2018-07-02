export CUDA_VISIBLE_DEVICES=$device
# Debug output
set -ex
name=imagenet_resnet_9blocks_attention_residual
python test.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_attcnn \
                --which_model_netG resnet_9blocks_attention_residual \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir '/private/home/zizhao/work/checkpoint_fmri' \
                --batchSize 64 \
                --input_nc 2 \
                --output_nc 2 \
                --how_many -1 \
                --non_strict_state_dict \
                --which_epoch 100

