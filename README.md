RL environment and training script for CacheSimulator
==============

***Note for this ma_covert version, which uses multiagent for covert channel discovery, there are several requirements***

wandb version (use an older version)
```
pip install wandb==0.13.0
```

rlmeta version, please follow RLmeta by jx_cui, see ```https://github.com/cuijiaxun/rlmeta``` (use ```marl``` branch).


## Normal steu

We use conda to manage all the python dependencies, we assume the ```conda``` is already installed, and we provide a script to install all the depedencies using ```conda```.

Creating a conda environment:

```
$ conda create --name ma_covert python=3.8
```
Then press enter when prompt.

Activate the conda environment

```
$ conda activate ma_covert
```
Undet the py38 environment

```
(ma_covert) $ pip install scikit-learn seaborn pyyaml hydra-core terminaltables pep517
```

The environment is based on openai [gym](https://github.com/openai/gym). To install it, use the following.

```
(ma_covert) $ pip install gym==0.25
```

Please follow the [PyTorch Get Started](https://pytorch.org/get-started/locally/) website to install PyTorch with proper CUDA version. One example is listed below.
```
(ma_covert) $ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
```

The enviroment needs [moolib](https://github.com/facebookresearch/moolib) as the RPC backend for distributed RL training. Please follow the moolib installation instructions.
We recommend building moolib from source with the following steps.

```
(ma_covert) $ git clone https://github.com/facebookresearch/moolib
(ma_covert) $ cd moolib
(ma_covert) $ git submodule sync && git submodule update --init --recursive
(ma_covert) $ pip install .
```

Install rlmeta


```
$ git clone https://github.com/cuijiaxun/rlmeta
$ cd rlmeta
$ git checkout marl
$ git submodule sync && git submodule update --init --recursive
$ pip install -e .
```
