# export CUDA_VISIBLE_DEVICES=$device

checkpoints_dir='/private/home/zizhao/work/checkpoint_fmri/vae_session'
set -ex

name=imagenet_ResCVAE_betaopim_ctxasres
python test.py --dataroot 'ImageNet' \
                --name $name \
                --model ft_caenn \
                --which_model_netG resnet_9blocks_attention_residual \
                --loadSize 144 \
                --fineSize 128 \
                --norm instance \
                --checkpoints_dir $checkpoints_dir \
                --batchSize 64 \
                --input_nc 2 \
                --output_nc 2 \
                --no_dropout \
                --how_many 1000 \
                --n_samples 100 \
                --num_blocks 2 \
                --depth 2 \
                --ctx_gen_det_skip False \
                --ctx_as_residual \
                --non_strict_state_dict \
                --gpu_ids $CUDA_VISIBLE_DEVICES \
                | tee $checkpoints_dir/$name/'eval_log.txt'

# name=imagenet_resnet_CVAE2_.0122_noskip_ctxvae
# python test.py --dataroot 'ImageNet' \
#                 --name $name \
#                 --model ft_cae2nn \
#                 --which_model_netG resnet_9blocks_attention_residual \
#                 --loadSize 144 \
#                 --fineSize 128 \
#                 --norm instance \
#                 --checkpoints_dir $checkpoints_dir \
#                 --batchSize 64 \
#                 --input_nc 2 \
#                 --output_nc 2 \
#                 --no_dropout \
#                 --how_many 1000 \
#                 --n_samples 100 \
#                 --num_blocks 2 \
#                 --depth 2 \
#                 --ctx_gen_det_skip False \
#                 --gpu_ids $CUDA_VISIBLE_DEVICES \
#                 | tee $checkpoints_dir/$name/'eval_log.txt'