# A Repo for Benchmarking News Recommendation on MIND

This repository offers comprehensive classes and utilities for designing->training->testing a News Recommendation Model on [MIND](https://msnews.github.io/) dataset.

## Environment
```
python=3.8.11
torch==1.9.1
```

## Instruction
```bash
cd /data/v-pezhang/Code/Document-Reduction/Code

# manager will email you the training result, so you should add your gmail and password as follows:
mkdir data/configs
echo "email = 'namespace.pt@gmail.com'" >> data/configs/email.py
echo "password = 'your password'" >> data/configs/email.py

# train a basic twotower model with CNN news encoder and LSTM user encoder
python twotower.py

# test
python twotower.py -m test
```

- specify `-p=/your/path/to/MIND/root`
- specify `-d=x` if you want to train/test on the `x`-th gpu
- specify `-ws=x` if you want to train/test with `x` gpus
- for more hyper parameters, run `python twotower.py --help`


## How to define your own model
1. design your model in `models/` referring to `TwoTower.py`
2. write a script in `./` referring to `twotower.py`


## Basic Introduction
### Preprocess
in `utils/MIND.py`
- tokenize
- generate cache

### Manager
in `utils/Manager.py`
- handle hyper parameters
- prepare dataloaders
- handle the training/evaluation process

### News Encoder
in `models/Enocders`
- CNN
- RNN
- Multi Head Attention
- One layer bert
- Pooling

### User Encoder
in `models/Enocders`
- LSTM
- MHA
- LSTUR
- Pooling