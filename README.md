# CLIP-Help-SimCLR

This repository includes a PyTorch implementation of the ICML 2023 paper  [On the Generalization of Multi-modal Contrastive Learning]() authored by Qi Zhang*, [Yifei Wang*](https://yifeiwang.me), [Yisen Wang](https://yisenwang.github.io/). 

In this repository, we consider four strategies for leveraging CLIP to help self-supervised contrastive learning with SimCLR.


| Method | Baseline (SimCLR)  | AddNewPositive | DropFalsePositive |DropFalseNegative |DropEasyNegative|
|-----------|:---------------:|:--------:|:---------:|:------------:|:-----------:|
| Linear Acc|  61.2      |   67.4 (+6.2)   |    61.8 (+0.6)   |     61.4 (+0.2)    |62.3 (+1.1)|






## Enviroment Setup


Create a python enviroment with the provided config file and [miniconda](https://docs.conda.io/en/latest/miniconda.html):

```(bash)
conda env create -f environment.yml
conda activate simclr_pytorch

export IMAGENET_PATH=... # If you have enough RAM using /dev/shm usually accelerates data loading time
export EXMAN_PATH=... # A path to logs
```


## Import CLIP Models

Install the official CLIP respository.

```
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

And we download the official [CLIP]("https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt") models.


## Training
Model training consists of two steps: (1) self-supervised encoder pretraining and (2) classifier learning with the encoder representations. Both steps are done with the `train.py` script. 

### Self-supervised pretraining


#### ImageNet/
The configs `imagenet_params_epochs*_bs*.yaml` contain the parameters to reproduce results for ImageNet dataset. The pretraining command is:

```(bash)
python train.py --config configs/imagenet_train_epochs100_bs512.yaml --method <Method>
```

The methods include 'simclr', 'new_positive', 'drop_false_positive', 'drop_false_negative', 'drop_easy_negative'.

### Linear Evaluation
To train a linear classifier on top of the pretrained encoder, run the following command:

```(bash)
python train.py --config configs/cifar_eval.yaml --encoder_ckpt <path-to-encoder>
```


## Citing this work


If you find our code useful, please cite
```
@inproceedings{
zhang2023on,
title={On the Generalization of Multi-modal Contrastive Learning},
author={Qi Zhang and Yifei Wang and Yisen Wang},
booktitle={International Conference on Machine Learning},
year={2023},
}
```
 


## Acknowledgements
Our codes borrows the implementations of SimCLR in https://github.com/AndrewAtanov/simclr-pytorch.
