AutoCAT
==============
This repo contains artifacts of the paper:

* "AutoCAT: Reinforcement Learning for Automated Exploration of cache-Timing Attacks" (HPCA'23).

You can find the paper at the [HPCA website](https://hpca-conf.org/2023/).

## Artifact contents

The artifact contains two parts

* CacheSimulator and PPO trainer

    * CacheSimulator is based on an [open source implementation](https://github.com/auxiliary/CacheSimulator) from [auxiliary](https://github.com/auxiliary).
    * PPO trainer is using [rlmeta](https://github.com/facebookresearch/rlmeta) from [Meta AI](https://ai.facebook.com).

* StealthyStreamline Attack code

## Test steup

We use conda to manage all the python dependencies, we assume the ```conda``` is already installed, and we provide a script to install all the depedencies using ```conda```. 

```
$ cd ${GIT_ROOT}/src/rllib
$ bash deploy_conda_rllib.sh
```

The environment is based on openai [gym](https://github.com/openai/gym). To install it, use the following.

```
$ pip install gym
```

The RL trainer is based on [RLMeta](https://github.com/facebookresearch/rlmeta). Please follow setup process on [rlmeta](https://github.com/facebookresearch/rlmeta) for install RLMeta. 

## General flow for Training and evaluating RL agent

After install rlmeta, you can launch the experiment to train the RL agent

```
$ cd ${GIT_ROOT}/src/rlmeta
$ python train_ppo_transformer.py
```
Use ```Ctrl+C``` to stop the training, which will save the checkpoint of the RL agent. To extract the pattern of the RL agent, use the following script

```
$ cd ${GIT_ROOT}/src/rlmeta
$ python sample.py
```
For several scenarios, training may take long time, to save the time of reviewers, we provide pretrained checkpoints and reviewers can sample it directly.


## Experiments

We provide scripts to reproduce the following appeared in the original paper.

* Table IV
* Table V
* Table VI
* Table VII
* Table VIII
* Table IX


## Repo Structure 

```
-configs            # this is the directory for CacheSimulotor configuration
-docs               # documentations
-env_test           # contains testing suit for simulator and replacement policy
-src
 |--config          # gym environment configurations
 |--cyclone_data    # data for training cyclone svm classifier
 |--fig             # positions for storing the figure
 |--models          # customized pytorch models for the RL agent to use
 |--rllib           # scripts for launching RLLib based experiments
 |--rlmeta          # scripts for launching RLMeta-basee experiments
 |--setup_scripts   # some scripts for setup the environment
 |--cache.py        # the cache logic implementation
 |--cache_simulator.py              # the interface of the cache simulator
 |--replacement_policy.py           # define the replacement policy for the cache
 |--cache_guessing_game_env_impl.py # the gym implementation of the cache
 |--cchunter_wrapper.py             # the wrapper that implements cchunter attack detector
 |--cyclone_wrapper.py              # the wrapper that implements cyclone attack detector
-third_party        # position for third-party libraries like 
-traces             # places for traces of CacheSimulator
```
## Contact

Please direct any questions to Mulong Luo ```ml2558@cornell.edu```.

## Research Paper

The paper is available in the procceedings the 29th Sympisum on High Performance Computer Architecture [(HPCA)](https://hpca-conf.org/2023/). Yo can cite our work with the bibtex entry

```
@inproceedings{luo2023autocat
year={2023},
title={{AutoCAT: Reinforcement Learning for Automated Explorations Cache-Timing Vulnerabilities}},
booktitle={29th Sympisum on High Performance Computer Architecture (HPCA)},
author={Mulong Luo and Wenjie Xiong and Geunbae Lee and Yueying Li and Xiaomeng Yang and Amy Zhang and Yuandong Tian and Hsien Hsin S. Lee and G. Edward Suh}
}
```

