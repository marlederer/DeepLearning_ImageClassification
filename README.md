# DeepLearning_ImageClassification

## Overview

Welcome to the Media Content Representation and Information Extraction exercise! The primary objective of this exercise is to immerse yourself in the significance of representation and extraction of information from intricate media content, specifically images or text. Throughout this exercise, you will be working with datasets designed for image classification purposes.

## Dataset

### German Traffic Signs Dataset

The German Traffic Signs dataset is a collection of images representing various traffic signs commonly found on roadways in Germany. Each image is labeled with the corresponding traffic sign class, making it an ideal dataset for tasks related to image classification and object recognition in the context of traffic signs.

Dataset Source: [German Traffic Signs Dataset]([https://benchmark.ini.rub.de/gtsdb_dataset.html](https://benchmark.ini.rub.de/gtsrb_dataset.html))

### CIFAR-10 Dataset

The CIFAR-10 dataset is a well-known benchmark dataset in the field of computer vision. It consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The dataset is divided into training and testing sets, making it suitable for various image classification tasks.

Dataset Source: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

## Goals

1. **Understanding Representation**: Gain insights into the importance of representing information in the context of complex media content.

2. **Information Extraction**: Explore techniques for extracting valuable information from images or text within the provided datasets.

## Getting Started

To get started with this exercise, follow these steps:

1. **Clone this Repository**: Begin by cloning this GitHub repository to your local machine.

```bash
git clone https://github.com/marlederer/DeepLearning_ImageClassification.git

```
2. Install Dependencies: Ensure that you have all the necessary dependencies installed. You can find the required libraries and tools in the `requirements.txt` file.
```bash
pip install -r requirements.txt
```
3. Execute the main.
```bash
cd code
python3 main.py --dataset 'cifar-10'
```
There are several parameters available. They are listed in the file `parse.py` 
