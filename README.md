  
  [![CircleCI](https://circleci.com/gh/facebookresearch/active-mri-acquisition/tree/master.svg?style=svg&circle-token=23a90ca66ff4c99cc0333b1f3ab46298bc5f3ec5)](https://circleci.com/gh/facebookresearch/active-mri-acquisition/tree/master) ![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)



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
Once your conda environment is activated, install [PyTorch](https://pytorch.org/) 
with the appropriate CUDA configuration for your system. 

To install  run

```bash
$ pip install pyxb==1.2.6
$ pip install -e .
```

If you also want the 

## Configuring the environment
To run the environments, you need to configure a couple of things. If you try to run any of the
default environments for the first time (for example, see our [intro notebook](https://github.com/facebookresearch/active-mri-acquisition/blob/master/notebooks/miccai_example.ipynb)), 
you will see a message asking you to add some entries to the `defaults.json` file. This file will
be created automatically the first time you run it, located at `$USER_HOME/.activemri/defaults.json`.
It will look like this:
```json
{
  "data_location": "",
  "saved_models_dir": ""
}
```
To run the environments, you need to fill these two entries. Entry `"data_location"` must point to 
the root folder in which you will store the fastMRI dataset (for instructions on how to download 
the dataset, please visit https://fastmri.med.nyu.edu/). Entry `"saved_models_dir"` indicates the 
folder where the environment will look for the checkpoints of reconstruction models.


# License

`active-mri-acquisition` is MIT licensed, as found in the [LICENSE file](https://github.com/facebookresearch/active-mri-acquisition/blob/master/LICENSE).