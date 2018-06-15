set -ex
name=imagent_pix2pix_insnorm_lsgan_resnet9
path='/private/home/zizhao/work/checkpoint_fmri' 
CUDA_VISIBLE_DEVICES=$device python test.py --dataroot 'ImageNet' \
                --name $name \
                --model pix2pix \
                --which_model_netG resnet_9blocks \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $path \
                --input_nc 1 \
                --output_nc 1 \
                --batchSize 64 \
                --how_many 10 \
                --which_direction BtoA \
                | tee -a $path/$name/'val_log.txt' \

# use BtoA direction to test whether the model is consistent