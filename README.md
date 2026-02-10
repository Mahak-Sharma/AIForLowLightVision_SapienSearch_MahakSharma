# AI For Low Light Vision

## Problem Statement:

Participants must enhance low light underwater images, track single underwater objects or segment underwater objects under challenging visual conditions and form an end-to-end vision pipeline. As a bonus task participants are expected to deploy their pipeline model on the cloud. Finally, shortlisted teams are expected to prepare a presentation for the judges

# TASK 1: Underwater Image Enhancement

## Objective

Develop a robust underwater image enhancement model to:
  Correct color distortion
  Reduce haze and scattering effects
  Improve contrast and sharpness
  Restore perceptual quality while preserving structural details

## Models Used:

RetinexFormer enhances images by combining Retinex theory with transformer-based global modeling.
The input image is first decomposed into reflectance (color and texture) and illumination (lighting and haze).
Each component is enhanced separately using transformer blocks to correct color distortion and improve brightness and contrast.
The enhanced components are then recombined to produce a visually improved image with better structural and perceptual quality.

## Setup

Using resources in TASK 1 folder.

### How to Run
#### 1). Create environment

`python -m venv venv`

`venv\Scripts\activate`

#### 2). Install Dependencies

`pip install -r requirements.txt`

#### 3). Run python training script

`python train.py`

#### 4). Run python prediction script

`python predict.py`

## OUTPUT

<img width="256" height="256" alt="img_0100" src="https://github.com/user-attachments/assets/11359e76-16f2-4873-b4bb-e4a100c8db88" />

Provided Image

<img width="256" height="256" alt="img_0100_result" src="https://github.com/user-attachments/assets/071e2e30-5f0e-43d0-a846-af6aa77655de" />

Enhanced Image

#### Evaluation Matrices Result

PSNR: 20.27

SSIM: 0.8279

LPIPS: 0.2550 

UCIQE: 71.3395

# TASK 2: Track B — Underwater Semantic Segmentation

## Objective

The objective of this track is to develop a robust underwater semantic segmentation model capable of accurately identifying and segmenting multiple classes in degraded underwater environments. Participants are required to design, train, and evaluate a deep learning-based segmentation model that can assign a class label to every pixel in an underwater image, enabling fine-grained scene understanding rather than just bounding-box detection.

## Models Used

In this project, SegFormer is used as a baseline transformer model to evaluate segmentation-aware feature extraction for underwater image enhancement.
Its hierarchical transformer encoder provides multi-scale contextual features, which are leveraged to understand scene structure and object boundaries.
The segmentation output and intermediate features help guide enhancement by preserving structural details while improving visual quality.

## Setup

Using resources in TASK 2 folder.

### How to Run
#### 1). Create environment

`python -m venv venv`

`venv\Scripts\activate`

#### 2). Install Dependencies

`pip install -r requirements.txt`

#### 3). Run python training script

`python train.py`

#### 4). Run python prediction script

`python predict.py`

## OUTPUT

![image_0001](https://github.com/user-attachments/assets/66aba627-681a-4f8a-9f74-fd2ad6e06a78)

Provided Image

![img result](https://github.com/user-attachments/assets/b0443cb9-5a10-49a9-a9bc-228175607947)

Segmented Image

#### Evaluation Matrices Result

mIOU: 81.76

F1: 73.24
