# Compare efforts of training with different replacement policies

First, go to the directory.

```
cd ${GIT_ROOT}/src/rlmeta
```

To train a config in Table V, use the following script:

```
$ python train_ppo_attack.py env_config=<NAME_OF_THE_CONFIG>
```

There are 17 configs in Table V, and we have ```hpca_ae_exp_5_1```, ```hpca_ae_exp_4_2```, ..., ```hpca_ae_exp_5_3``` correpondingly, replace ```<NAME_OF_THE_CONFIG>``` with these.

Use ```Ctrl+C``` to interrupt the training, which will save a checkpoint in the given path.

To extract the attack pattern from the checkpoint, use the following command (replace ```<NAME_OF_THE_CONFIG>``` and ```<ABSOLUTE_PATH_TO_CHECKPOINT>```) correspondingly.

```
$ python sample_attack.py  env_config=<NAME_OF_THE_CONFIG> checkpoint=<ABSOLUTE_PATH_TO_CHECKPOINT>
```

Since the training takes some time, we provide pretrained checkpoints in the following directory ```checkpoint```. Use the above command to replay these checkpoints.

For example

```
$ python sample_attack.py  env_config=hpca_ae_exp_4_1 checkpoint=${GIT_ROOT}/src/rlmeta/data/table4/hpca_ae_exp_4_1/ppoagent.pth
```

will print out the following: