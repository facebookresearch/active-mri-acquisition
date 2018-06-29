
name=imagenet_unet_vae
python test.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_vaenn \
                --which_model_netG jure_unet_vae_residual \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir '/private/home/zizhao/work/checkpoint_fmri' \
                --input_nc 2 \
                --output_nc 1 \
                --batchSize 64 \
                --how_many 256 \
