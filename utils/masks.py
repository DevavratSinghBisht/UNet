import logging
import argparse
import numpy as np
from functools import partial
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from tqdm import tqdm

from images import load_image

def unique_mask_values(idx: str, mask_dir: Path, mask_suffix:str) -> np.ndarray:
    """
        Returns an array of unique mask vlaues.
        Ideally returns [0,1].

        params:
            idx         (str)           : mask index eg. 31186febd775_09
            mask_dir    (WindowsPath)   : path to mask directory eg. data\masks
            mask_suffix (str)           : mask suffix name eg. _mask

        return
            mask    (np.ndarray)        : array of unique mask values
    """
    
    # filters the file
    mask_files = list(mask_dir.glob(idx + mask_suffix + '.*'))
    assert len(mask_files) == 1 , f"Number of mask files should be one, found {len(mask_files)}"
    mask_file = mask_files[0]
    
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')
    


def verify_mask_values(images_dir: Path, mask_dir: Path, mask_suffix: str = '_mask'):

    logging.info('Scanning mask files to determine unique values')
    ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        
    with Pool() as p:
        unique = list(tqdm(
            p.imap(partial(unique_mask_values, mask_dir=mask_dir, mask_suffix=mask_suffix), ids),
            total=len(ids)
        ))

    mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
    logging.info(f'Unique mask values: {mask_values}')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    
    parser.add_argument('--train-imgs', '-ti', type=str, default='data/train/images', help='Path to the directory containing training images.')
    parser.add_argument('--train-masks', '-tm',type=str, default='data/train/masks', help='Path to the directory containing training masks.')
    parser.add_argument('--val-imgs', '-vi', type=str, default='data/val/images', help='Path to the directory containing validation images.')
    parser.add_argument('--val-masks', '-vm', type=str, default='data/val/masks', help='Path to the directory containing validation masks.')
    parser.add_argument('--mask-suffix', '-sckpt', type=str, default='_mask', help='mask suffix name eg. _mask')
    
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    logging.info("Verifying Training Masks.")
    verify_mask_values(Path(args.train_imgs), Path(args.train_masks), args.mask_suffix)
    logging.info("Training Mask verification complete.")

    logging.info("Verifying Validation Masks.")
    verify_mask_values(Path(args.val_imgs), Path(args.val_masks), args.mask_suffix)
    logging.info("Validation Mask verification complete.")