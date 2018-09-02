echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
set -ex

root='/private/home/zizhao/work/checkpoint_fmri/resnset128_scratch/'
mkdir -p root
python main.py -a resnet50 '/datasets01/imagenet_resized_144px/060718/061417' -b 256 --gpu_ids $CUDA_VISIBLE_DEVICES --save_path $root'checkpoint.pth.tar'


# python main.py -a resnet50 '/datasets01/imagenet_resized_144px/060718/061417' --pretrained -b 256 --gpu_ids $CUDA_VISIBLE_DEVICES --save_path $root'checkpoint.pth.tar'