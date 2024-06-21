# Semantic Image Segmentation using U-Net

Implementation of U-Net for Image Segmentation.

## Quick Start

### Without Docker

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch](https://pytorch.org/get-started/locally/)

3. Install Dependencies 
```bash 
pip install -r requirements.txt
```

4. Download the data

5. You can either place respecitve data in the following directories or use parameters to specify them.

Train Images    : `data/train/images`
Train Masks     : `data/train/masks`
Validation Images   : `data/val/images`
Validation Masks    : `data/val/masks`