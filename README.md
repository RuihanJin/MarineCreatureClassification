# Marine Creature Classfication

## Description

This project is my course project for "An introduction to Artificial Intelligence"

## Setup

The code relies on the deep learning framework of Pytorch

### Requirements

- Anaconda
- numpy
- torch
- torchvision
- tqdm 
- matplotlib
- Pillow
- bidict
- torchcam

## Dataset

Please download the dataset from [here](https://cloud.tsinghua.edu.cn/f/15dfcd0d51b6411ba3f4/?dl=1) and unzip it. You are recommand to rename the dataset directory into `marine-creature-dataset`.

## Pre-Trained Models

You can download the pretrained model from [here](https://cloud.tsinghua.edu.cn/d/10ea7a50c6654ef3b2bc/).

You are recommanded to put the pretrained model under the  `./pretrained_models/` directory.

## Train

Run the following command to train on ConvNet:

```
python train.py --epochs 100 --image_dir /path/to/your/dataset
```

Or you can train on ResNet101 or DenseNet121 with the folowing command:

```
python train.py --model resnet101 --epochs 20 --image_dir /path/to/your/dataset
```
```
python train.py --model densenet --epochs 20 --image_dir /path/to/your/dataset
```
During training, you can supervise the process with tensorboard. The best model on validation set will be recorded under the `./logs` directory.

Please refer to the `options.py` for detailed parameters of training.

## Inference

Run the following command to inference on trained model:

```
python inference.py --pretrained_model /path/to/your/model --image_dir /path/to/your/dataset
```

For example:

```
python inference.py --pretrained_model ./pretrained_models/densenet.pth --image_dir ./marine-creature-dataset
```

You can specifify network layer for creating the heatmap by adding the `--layer` options like:

```
python inference.py --pretrained_model ./pretrained_models/densenet.pth --image_dir ./marine-creature-dataset --layer net.features
```

After inference, you can see the classfication precision and recall value on terminal. The visualization output will be recorded under the `./results` directory.

Also, please refer to the `options.py` for detailed parameters of inferencing.
