# Reinforcement learning environment for Active MRI Acquisition

A Reinforcement learning environment to facilitate research on 
[active MRI acquisition](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Reducing_Uncertainty_in_Undersampled_MRI_Reconstruction_With_Active_Acquisition_CVPR_2019_paper.pdf). 
The goal of `active-mri-acquisition` is to provide a convenient gym-like interface to test
the use of reinforcement learning and planning algorithms for subject-specific acquisition 
sequences of MRI scans. 

This repository also contains scripts to replicate the experiments performed in

 
*Luis Pineda, Sumana Basu, Adriana Romero, Roberto Calandra, Michal Drozdzal, 
"Active MR k-space Sampling with Reinforcement Learning". MICCAI 2020.*

# Getting started

## Installation
We recommend creating a dedicated Python environment for this project, for instance by running
```bash
$ conda create --name active_mri python=3.7
$ conda activate active_mri
```
Once your conda environment is activated, make sure you have [PyTorch](https://pytorch.org/) 
installed with the appropriate CUDA version. For instance, by running 
(see PyTorch's website for right command for your system)

```bash
$ pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

Afterwards, run
```bash
$ pip install pyxb==1.2.6
$ pip install -e .
```