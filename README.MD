# Semantic Image Segmentation using U-Net

Implementation of U-Net for Image Segmentation.

## Quick Start

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch](https://pytorch.org/get-started/locally/)

3. Install Dependencies 
```bash 
pip install -r requirements.txt
```

4. Download the data
    * You can use the `download_carvana_data.bat` or the `download_carvana_data.sh` file located in the `scripts` directory to download [Carvana Image Masking Challenge Dataset](https://www.kaggle.com/c/carvana-image-masking-challenge)
    * After you have downloaded the data you can create the validation set using the `create_carvana_validation.py`. You can use `--val-pct` to specify the percentage of data you want to use for validaton. By default the script uses 10% of the data for validation.
    example:
    ```bash
    example command: python create_carvana_validation.py --val-pct 20
    ```

OR

4. Use your own data
   
   If you want to use the your own directory structure you can also pass the path to your directories using the following flags:

    * --train-images: path to training images
    * --train-masks : path to training masks
    * --val-images  : path to validation images
    * --val-masks   : path to validation masks

    ```bash
    sample command: python train.py --train-images data/train/images --train-masks data/train/masks --val-images data/val/images --val-masks data/val/masks
    ```


## Default Arguments for the Training Script    
Following are the arguments that the train script takes along with their default values:

| Parameter       | Short Notation | Default Value     | Data Type   | Description                                               |
| --------------- | -------------- | ----------------- | ----------- | --------------------------------------------------------- |
| --train-images  | ti             | data/train/images | str         | Path to the directory containing training images.         |
| --train-masks   | tm             | data/train/masks  | str         | Path to the directory containing training masks.          |
| --train-masks   | vi             | data/val/images   | str         | Path to the directory containing validation images.       |
| --val-masks     | vm             | data/val/masks    | str         | Path to the directory containing validation masks.        |
| --save-ckpt     | svckpt         | checkpoints       | str         | Path to the save model checkpoint.                        |
| load-ckpt       | ldckpt         | False             | bool or str | Path to the model checkpoint. Load model from a .pth file |
| epochs          | e              | 5                 | int         | Number of epochs                                          |
| --batch-size    | b              | 1                 | int         | Batch size                                                |
| --learning-rate | lr             | 1e-5              | float       | Learning Rate                                             |
| --scale         | s              | 0.25              | float       | Downscaling factor of the images                          |
| --amp           | -a             | False             | bool        | Use mixed precision                                       |
| --bilinear      | -bil           | False             | bool        | Use bilinear upsampling                                   |
| --classes       | c              | 1                 | int         | Number of classes                                         |


## Citations

### U-Net
U-Net paper was authored by Olaf Ronneberger, Philipp Fischer, Thomas Brox:

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

![U-Net network architecture](https://i.imgur.com/jeDVpqF.png)


### Carvana Datset

[Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge/overview)

![input and output for a random image in the test dataset](https://i.imgur.com/GD8FcB7.png)


<!-- ## Sample Results -->
<!-- TODO add sample results -->