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

## SETUP

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

Enhanced Iamge

#### Evaluation Matrices Result
PSNR: 20.27
SSIM: 0.8279
LPIPS: 0.2550 
UCIQE: 71.3395
