# maua-style

Based on this excellent implementation https://github.com/ProGamerGov/neural-style-pt

Video style transfer and mostly broken support for videos as style has been added.

# Installation

## CUDA

First you need to have a working [CUDA toolkit installation](https://developer.nvidia.com/cuda-downloads).

An easy option to get a working CUDA environment on Ubuntu is [Lambda Stack](https://lambdalabs.com/lambda-stack-deep-learning-software).

Alternatively, I personally use [Anaconda](https://docs.anaconda.com/anaconda/install/index.html):
```bash
conda install -c conda-forge cudatoolkit-dev cudatoolkit cudnn
```
(this lets you use different CUDA toolkits for different environments)

## Maua Style

Whatever way you've set up your CUDA environment, once it's working, run the following to install this repository and its requirements:
```bash
git clone --recursive https://github.com/JCBrouwer/maua-style.git
cd maua-style
pip install -r requirements.txt
```
(this can take quite a while as Cupy needs to build its wheel from scratch, at least on my system)

# Usage

## Image style transfer

## Video style transfer

## CLIP + VQGAN style transfer

## Videos as style

# Credits

For optical flow support this repo includes a collection of excellent optical flow model reproductions by Simon Niklaus (@sniklaus).
Each repository has its own license and terms. Make sure to read and adhere to them as well as cite accordingly when using optical flow.

