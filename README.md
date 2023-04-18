# MACTA

> [**A Multi-agent Reinforcement Learning Approach for Cache Timing Attacks and Detection**](https://openreview.net/forum?id=CDlHZ78-Xzi)\
> Jiaxun Cui, Xiaomeng Yang*, Mulong Luo*, Geunbae Lee*, Peter Stone, Hsien-Hsin S. Lee, Benjamin Lee, Edward Suh, Wenjie Xiong^, Yuandong Tian^\
> International Conference on Learning Representations (_ICLR 2023_)\
> \*Equal Second-author Contribution, ^Equal Supervising

[Paper](https://openreview.net/pdf?id=CDlHZ78-Xzi) | [Website]() | [Bibtex](#citation)

## Installation
```
conda create -n macta python=3.7
conda activate macta
pip install -r requirements.txt
# Install PyTorch
# Install RLMeta
```

## Quick Start with Pre-trained Models
We provide pretrained models of all methods, checkout the `checkpoints/`.
To run our pretrained model, simply modify the path to the checkpoints and run
```
cd src/rlmeta/macta
conda activate macta
python sample_multiagent.py
```

## Benign Trace Generation
If you want to use [SPEC 2017](https://www.spec.org/cpu2017/), please make sure you have liscence to it and follow the [instructions here](https://code.vt.edu/bearhw-public/rl-mem-trace) to generate the traces. To use the traces, specify the path to the trace files in the configs. You can also test some open-source datasets as well.

## Training
To train MACTA
```
cd src/rlmeta/macta
conda activate macta
python train/train_macta.py
```

## Evaluation
Please specify the agents and evaluation parameters the config in `src/rlmeta/macta/config/sample_multiagent.yml`
```
cd src/rlmeta/macta
conda activate macta
python sample_multiagent.py
```

## Citation
```bibtex
@inproceedings{cui2023macta,
    title = {A Multi-agent Reinforcement Learning Approach for Cache Timing Attacks and Detection},
    author = {Jiaxun Cui, Xiaomeng Yang, Mulong Luo, Geunbae Lee, Peter Stone, Hsien-Hsin S. Lee, Benjamin Lee, Edward Suh, Wenjie Xiong, Yuandong Tian},
    booktitle = {International Conference on Learning Representations (ICLR)},
    year = {2023}
}
```
