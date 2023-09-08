RL environment and training script for CacheSimulator
==============

Note for this ma_covert version, which uses multiagent for covert channel discovery, there are several requirements

wandb version (use an older version)
```
pip install wandb==0.13.0
```

rlmeta version, please follow RLmeta by jx_cui, see ```https://github.com/cuijiaxun/rlmeta``` (use ```marl``` branch).


This repo contains wrapper for the CacheSimulator environment based on 
https://github.com/auxiliary/CacheSimulator

For detailed description of CacheSimulator, see the original repo.

The environment is based on openai gym

```
$ pip install gym
```

The trainer is based on RLlib

```
$ pip install rllib
```

To run the training

```
$ cd src
$ python run_gym_rllib.py
```

## Set up Enviroment on a GPU machine

We use conda to manage all the python dependencies, we assume the ```conda``` is already installed, and we provide a script to install all the depedencies using ```conda```.

Creating a conda environment:

```
$ conda create --name py38 python=3.8
```
Then press enter when prompt.

Activate the conda environment

```
$ conda activate py38
```
Undet the py38 environment

```
(py38) $ pip install scikit-learn seaborn pyyaml hydra-core terminaltables pep517
```

The environment is based on openai [gym](https://github.com/openai/gym). To install it, use the following.

```
(py38) $ pip install gym
```

Please follow the [PyTorch Get Started](https://pytorch.org/get-started/locally/) website to install PyTorch with proper CUDA version. One example is listed below.
```
(py38) $ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
```

The enviroment needs [moolib](https://github.com/facebookresearch/moolib) as the RPC backend for distributed RL training. Please follow the moolib installation instructions.
We recommend building moolib from source with the following steps.

```
(py38) $ git clone https://github.com/facebookresearch/moolib
(py38) $ cd moolib
(py38) $ git submodule sync && git submodule update --init --recursive
(py38) $ pip install .
```
