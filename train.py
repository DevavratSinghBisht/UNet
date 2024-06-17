import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from evaluate import evaluate
from unet import UNet
from dataset import CarvanaDataset
from dice_score import dice_loss


def train(
        model:              UNet,
        train_imgs:         str,
        train_masks:        str,
        val_imgs:           str,
        val_masks:          str,
        device:             torch.device,
        epochs:             int = 5,
        batch_size:         int = 1,
        lr:                 float = 1e-5,
        img_scale:          float = 0.25,
        amp:                bool = False,
        weight_decay:       float = 1e-8,
        momentum:           float =0.999,
        gradient_clipping:  float = 1.0,
        save_ckpt:          bool | str = False
) -> UNet:
    
    '''
    Trains a Image Segmentation Model

    params:
        model :             Model to be trained
        train_imgs:         Path to the directory containing input images for training
        train_masks:        Path to the directory containing training target mask images
        val_imgs:           Path to the directory containing input images for validation
        val_masks:          Path to the directory containing validation target mask images
        device:             Device to process tensor on, usualy cpu or gpu
        epoch:              Number of epochs to train the model
        batch_size:         Batch size for the dataset
        lr:                 Learning Rate for training
        img_scale:          Scaling factor for the image. It should be in the range of [0, 1] both inclusive
        amp:                If true then mixed precision is used
        weight_decay:       Weight Decay for optimizer
        momentum:           Momentum for the optimizer
        gradient_clipping:  Value to clip the gradient on
        save_ckpt:          If string is provided then the checkpoint is saved in that directory 
                            else if it is False then checkpoint is not saved

    returns:
        model, history :    Trained model object along with history contating the training and validation metrics. 
    '''
    
    # Creating a list of dictionay to save metrics
    train_history = []
    val_history = []


    # Create dataset
    train_set = CarvanaDataset(train_imgs, train_masks, img_scale)
    val_set = CarvanaDataset(val_imgs, val_masks, img_scale)

    # split train and val
    n_val = len(train_set)
    n_train = len(val_set)
    # train_set, val_set = random_split(dataset,
    #                                 [n_train, n_val],
    #                                 generator=torch.Generator().manual_seed(seed))
    
    # Creating dataloader
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    # Initalize Logging

    logging.info(f''' Starting Training:
        Train Images:       {train_imgs}
        Train Masks:        {train_masks}
        Validation Images:  {val_imgs}
        Validation Masks:   {val_masks}
        Epochs:             {epochs}
        Batch size:         {batch_size}
        Learning rate:      {lr}
        Training size:      {n_train}
        Validation size:    {n_val}
        Checkpoints:        {save_ckpt}
        Device:             {device.type}
        Images scaling:     {img_scale}
        Mixed Precision:    {amp}
        Wight Decay:        {weight_decay}
        Momentum:           {momentum}
        Gradient Clipping:  {gradient_clipping}
    ''')

    # Set up Optimizers, Loss, LR Scehduler and loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                           lr=lr,
                           weight_decay=weight_decay,
                           momentum=momentum,
                           foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps'else 'cpu', enabled=amp):
                    pred_masks = model(images)
                    if model.n_classes == 1:
                        loss = criterion(pred_masks, true_masks.float())
                        loss += dice_loss(F.sigmoid(pred_masks.squeeze(1)), true_masks.squeeze(1).float(), multiclass=False)

                    else:
                        # Multi Class
                        # TODO causes error, solve them
                        loss = criterion(pred_masks, true_masks)
                        # TODO add dice_loss here
                        # loss += dice_loss(
                        #     F.softmax(pred_masks, dim=1).float(),
                        #     F.one_hot(true_masks, model.n_classes).permute(0, 3, 1 ,2).float(),
                        #     multiclass=True
                        # )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                train_history.append({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # TODO Evaluation
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        
                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        
                        val_history.append({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': images[0].cpu(),
                            'masks': {
                                'true': true_masks[0].float().cpu(),
                                'pred': pred_masks.argmax(dim=1)[0].float().cpu(),
                            },
                            'step': global_step,
                            'epoch': epoch
                        })

        if save_ckpt:
            save_ckpt = Path(save_ckpt)
            save_ckpt.mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(save_ckpt / f'checkpoint_epoch{epoch}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')

    history = (train_history, val_history)
    return model, history 


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    
    parser.add_argument('--train-imgs', '-ti', type=str, default='data/train/images', help='Path to the directory containing training images.')
    parser.add_argument('--train-masks', '-tm',type=str, default='data/train/masks', help='Path to the directory containing training masks.')
    parser.add_argument('--val-imgs', '-vi', type=str, default='data/val/images', help='Path to the directory containing validation images.')
    parser.add_argument('--val-masks', '-vm', type=str, default='data/val/masks', help='Path to the directory containing validation masks.')
    parser.add_argument('--save-ckpt', '-sckpt', type=str, default='checkpoints', help='Path to the save model checkpoint.')
    parser.add_argument('--load-ckpt', '-lckpt', type=str, default=False, help='Path to the model checkpoint. Load model from a .pth file')

    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-lr', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    
    parser.add_argument('--scale', '-s', type=float, default=0.25, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    
    return parser.parse_args()

if __name__ == "__main__":

    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')
    
    if args.load_ckpt:
        state_dict = torch.load(args.load_ckpt, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load_ckpt}')
    
    model.to(device=device)
    
    try:
        model, history = train(
            model=model,
            train_imgs = args.train_imgs,
            train_masks = args.train_masks,
            val_imgs = args.val_imgs,
            val_masks = args.val_masks,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            img_scale=args.scale,
            amp=args.amp,
            save_ckpt=args.save_ckpt
        )

        # TODO pretty print history
        # save history in json, use a flag input arg

    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Reduce memory to continue training.'
                      'Consider following options and try training again:'
                      'Reducing batch size (--batch-size)'
                      'Reducing the scaleing factor (--scale), this will reduce the size of images.'
                      'Enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()        