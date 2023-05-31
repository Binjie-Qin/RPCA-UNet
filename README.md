# RPCA-UNet

By Binjie Qin, Haohao Mao, Yiming Liu, Jun Zhao, Yisong Lv, Yueqi Zhu, Song Ding, and Xu Chen

This repository is a pytorch implementation of ["Robust PCA Unrolling Network for Super-Resolution Vessel Extraction in X-Ray Coronary Angiography"](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9780367).

## Introduction
We propose a novel robust PCA unrolling network with sparse feature selection for super-resolution XCA vessel imaging. Being embedded within a patch-wise spatiotemporal super-resolution framework that is built upon a pooling layer and a convolutional long short-term memory network, the proposed network can not only gradually prune complex vessel-like artefacts and noisy backgrounds in XCA during network training but also iteratively learn and select the high-level spatiotemporal semantic information of moving contrast agents flowing in the XCA-imaged vessels.

## Environment

We could ensure that the code is available in such environment.
- RTX 3090 with driver version 470.141.03
- CUDA 11.1
- cuDNN 8.0.4
- conda create -n RPCA python=3.7.0
- source activate RPCA
- pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
- pip install matplotlib
- pip install tqdm
- pip install scipy
- pip install opencv-python
- pip install scikit-learn
- pip install scikit-image

## Data Preparation
Put your images(origin images, vessel images, background images) in '.\Data\rawData'. The folder structure should look like
```
The project
|
└───Data
|   └───rawData
|   |   └───51-32-1(sequence name)
|   |   |   └───ori_1.png
|   |   |   └───ori_2.png
|   |   |   └───...
|   |   |   └───vessel_1.png
|   |   |   └───vessel_2.png
|   |   |   └───...
|   |   |   └───background_1.png
|   |   |   └───background_2.png
|   |   |   └───...
|   |   └───103-25-1
|   |   └───...
|   └───processedData
└───Data_test
|   └───rawData_test
|   |   └───403-16-1
|   |   |   └───...
└───classes
└───network
└───Results
└───...
```
(Since the dataset is too large to be uploaded to github, Please email the author bjqin@sjtu.edu.cn for the access of more dataset.)

## Getting Started

1. Run `rawDataProcess.py` to generate image patches for training.
The output will be saved in `.\Data\processedData`.

2. Run `train.py` to train the network.
The model will be saved in `.\Results`.

3. Run `batch_test.py` to obtain the vessel extraction result.
The result will be saved in `.\Results\test`.
   
4. Run `batch_cal_metric.py` to calculate the metrics.
