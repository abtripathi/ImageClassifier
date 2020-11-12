import matplotlib.pyplot as plt
import numpy as np
import json
import os.path

## this is the helper class

def print_results(**kwargs):
    
    for key,value in kwargs.items():
        print(key+" =", value,end='\t'*2)
    print()
    
    
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    if title:
        ax.set_title(title)
    
    return ax
   
def label_mapping(filename):
             
    with open(filename, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
    
