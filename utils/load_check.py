import torch
import numpy as np

def load_check(path):
    checkpoint = torch.load(path)
    # print(checkpoint['channels'])
    if 'state_dict' in checkpoint.keys():
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint.keys():
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if 'channels' in checkpoint.keys() and checkpoint['channels'] is not None:
        channels = checkpoint['channels']
        print('channels:', channels)
    else:
        channels = None
    if 'heads' in checkpoint.keys() and checkpoint['heads'] is not None:
        heads = checkpoint['heads']
        print(heads)
    else:
        heads = None
    return state_dict, channels, heads