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
`active-mri-acquisition` has been tested on Python 3.6.10. To install dependencies run 
the following command:

```bash
$ conda env create -f environment.yml
```

## Using the environment
The following code creates an environment
```python
import gym
import rl_env
env = gym.make("ActiveMRIAcquisition-v0")

```