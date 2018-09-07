# fast MRI project

This repo provides the source code to reproduce the proposed method for undersampled MRI reconstruction, uncertainty measurement, and k-space acqusition planning.

## Models
The key code files include:
- ```models/netwokrs.py```: network definition 
- ```models/ft_pasgan_model.py```: management the training and evaluation
- ```models/ft_pasgan_raw_model.py```: management the training and evaluation on raw k-space data (it will merged with ft_pasgan_model.py later on)
- ```data/ft_data_loader/ft_data_loader.py```: data loader 
(Currently the repo still contain a lot of redundant code, which was used for different experiments and trials. Will cleanup later on)

## Training
Train the full model 

    CUDA_VISIBLE_DEVICES=0,1 sh knee_scripts/train_ftpasgan.sh

## Pretrained models
Pre-trained models are provided so you can directly do evaluation to produce the results in the presentation slide.
The checkpoint path is ```/private/home/zizhao/work/checkpoint_fmri/mri_zz_models```

## Evaluation
Evaluate the model to simulate k-space acqusition planning demos

    CUDA_VISIBLE_DEVICES=0,1 sh knee_scripts/test_recommend_ftpasnet.sh

Evaluate the model to generate sample results

    CUDA_VISIBLE_DEVICES=0,1 sh knee_scripts/sampling_ftpasgan.sh

Evaluate the model to compute metrics overall the whole val set

    CUDA_VISIBLE_DEVICES=0,1 sh knee_scripts/test_ftpasgan.sh

## Visualiztion
Build a ssh tunnel from cluster port 8888 to localhost:8888 and goto the checkpoint folder, e.g. ```/private/home/zizhao/work/checkpoint_fmri/mri_zz_models```.
Launch a simple http server

    python -m http.server 8888

Open in the browser

    http://localhost:8888/mri_zz_models/knee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm/

Besides the checkpoints, you will see three folders (if you run all three scripts in Evaluation). Open one of them to see resuls of the corresponding evaluation stuff. For example, open ```test_recommend_latest/``` to see the k-space acqusition planning demo.
 

## Data
- Training on the DICOM dataset needs the loader of the fast_mri repo. See ```data/ft_data_loader/ft_data_loader.py``` to set the library path. 
- Training on the raw k-space data needs the mmap data files at ```'/private/home/zizhao/work/mri_data/multicoil/raw_mmap/FBAI_Knee/'``` to load the KNEE_RAW data. It is generated using the code at ```https://github.com/facebookexternal/fast_mri/tree/master/experimental/anuroop/raw```.

## Acknowledgement
The code framework is inspired by the CycleGAN offcical [repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/pix2pix_model.py).

## Author
Zizhao Zhang\
Michal Drozdzal\
Adriana Romero Soriano
