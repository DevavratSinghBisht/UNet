import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6) -> Tensor:
    '''
    Calculated dice coefficient used to evaluate how god an image segmentation model is.

    params:
        input               : predicted mask
        target              : ground truth mask
        reduce_batch_first  : 
        epsilon             : small number to avoid division by 0 error

    returns:
        dice                : dice score
    '''


    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6) -> Tensor:
    '''
    Calculates dice coefficient in case of multiple classes by taking average of all classes.
    
    params:
        input               : predicted mask
        target              : ground truth mask
        reduce_batch_first  : 
        epsilon             : small number to avoid division by 0 error

    returns:
        dice                : dice score
    '''
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    '''
    Calculates dice loss
    Dice coefficient ranges from 0 to 1
    Dice loss is 1 - dice coefficient
    Dice Loss can be used as an objective to be minimized.

    params:
        input               : predicted mask
        target              : ground truth mask
        reduce_batch_first  : 
        epsilon             : small number to avoid division by 0 error

    returns:
        dice_loss           : dice loss
    '''
    
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    dice_loss = 1 - fn(input, target, reduce_batch_first=True)
    return dice_loss