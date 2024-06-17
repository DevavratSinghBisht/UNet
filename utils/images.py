import torch
import numpy as np
from PIL import Image
from os.path import splitext



def load_image(filename):

    # Split path and extension
    # path is anything before the extension 
    ext = splitext(filename)[1]

    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)