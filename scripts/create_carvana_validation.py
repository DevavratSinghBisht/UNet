# This script is used to create validation set for carvana dataset
# Make sure you have downloaded the train set before you run this script
# tain imags should be located at data/train/images
# train masks should be located at data/train/masks
# You can use download_carvana_dataset.bat or download_carvana_dataset.sh to download the dataset and place them in appropriate directories

import os
from os.path import splitext
from pathlib import Path
import random

import argparse

TRAIN_IMAGES_DIR = Path('data/train/images')
TRAIN_MASKS_DIR = Path('data/train/masks')
VAL_IMAGES_DIR = Path('data/val/images')
VAL_MASKS_DIR = Path('data/val/masks')
MASK_SUFFIX = '_mask'

def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='Create validation set for Carvana Dataset')
    parser.add_argument('--val-pct', '-v', type=float, default=10.0, help='Percentage of validation split.')

    return parser.parse_args()
    

if __name__ == "__main__":

    # get args
    args = get_args()
    
    # list all the train images
    assert TRAIN_IMAGES_DIR.exists() ,  "Training Image Directory does not exist,\
                                        make sure you have crated it \
                                        and placed it at data/train/images"
    assert TRAIN_MASKS_DIR.exists() ,   "Training Mask Directory does not exist,\
                                        make sure you have crated it \
                                        and placed it at data/mask/images"
    
    if VAL_IMAGES_DIR.exists():
        assert len(os.listdir(VAL_IMAGES_DIR)) == 0 , f'Validation Image Directory ({VAL_IMAGES_DIR}) contains data.'
    
    if VAL_MASKS_DIR.exists():
        assert len(os.listdir(VAL_MASKS_DIR)) == 0 , f'Validation Masks Directory ({VAL_MASKS_DIR}) contains data.'
  
    images = os.listdir(TRAIN_IMAGES_DIR)
    
    n_images = len(images)
    n_val_images = int((n_images * args.val_pct) // 100)
    print(n_images, n_images - n_val_images, n_val_images)
    # n_train_images = images - n_val_images

    # train_images = random.sample(images, n_train_images)
    val_images = random.sample(images, n_val_images)

    VAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    VAL_MASKS_DIR.mkdir(parents=True, exist_ok=True)

    for img in val_images:

        img_split = splitext(img)
        mask =  img_split[0] + MASK_SUFFIX + '.gif'
        
        os.rename(TRAIN_IMAGES_DIR/img, VAL_IMAGES_DIR/img)
        os.rename(TRAIN_MASKS_DIR/mask, VAL_MASKS_DIR/mask)
    