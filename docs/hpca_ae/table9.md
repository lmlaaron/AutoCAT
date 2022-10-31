# Table IX: bit rate, guess accuarcy and detection rate for attacks bypassing SVM-based detector

We compare the attack patterns found in Table VIII and epochs need for different replacement policies.


![](../../fig/table9.png)

First, go to the directory.

```
cd ${GIT_ROOT}/src/rlmeta
```

To train a config in Table V, use the following script:

```
$ python train_ppo_attack.py env_config=<NAME_OF_THE_CONFIG>
```

There are 17 configs in Table V, and we have ```hpca_ae_exp_5_1```, ```hpca_ae_exp_5_2```, ..., ```hpca_ae_exp_5_3``` correpondingly, replace ```<NAME_OF_THE_CONFIG>``` with these.

Use ```Ctrl+C``` to interrupt the training, which will save a checkpoint in the given path.

To extract the attack pattern from the checkpoint, use the following command (replace ```<NAME_OF_THE_CONFIG>``` and ```<ABSOLUTE_PATH_TO_CHECKPOINT>```) correspondingly.

```
$ python sample_attack.py  env_config=<NAME_OF_THE_CONFIG> checkpoint=<ABSOLUTE_PATH_TO_CHECKPOINT>
```

Since the training takes some time, we provide pretrained checkpoints in the following directory ```checkpoint```. 

To calculate the bit rate, max autocorrelation and accuracy of these scenarios, use the following.

```
```