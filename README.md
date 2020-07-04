# NeuralUCB
This repository contains our pytorch implementation of NeuralUCB in the paper [Neural Contextual Bandits with UCB-based Exploration](https://arxiv.org/pdf/1911.04462.pdf) (accepted by ICML 2020). 

## Prerequisites: 
* Pytorch and CUDA
* future==0.18.2
* joblib==0.15.1
* numpy==1.18.1
* pkg-resources==0.0.0
* scikit-learn==0.22.1
* scipy==1.4.1
* torch==1.5.0

## Usage:
Use python to run train.py for experiments.

## Command Line Arguments:
* --size: bandit algorithm time horizon
* --dataset: datasets
* --shuffle: to shuffle the dataset or not
* --seed: random seed for shuffle
* --nu: nu for control variance
* --lambda: lambda for regularization
* --hidden: network hidden size


## Usage Examples:
* Run experiments on [mnist](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf):
```bash
  -  python3 train.py --nu 0.00001 --lamdba 0.00001 --dataset mnist
```
