import os
import torch
from torch import nn
from model import VisionTransformer, SelfAttention
from config import get_train_config, get_eval_config
from checkpoint import load_checkpoint
from data_loaders import *
from utils import setup_device

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def compute_param(model):

    # print("create model")
    total_param = 0
    mlp_param = 0
    attn_param = 0
    for name, param in model.named_parameters():
        s_param = param.reshape(-1)

        if 'mlp' in name:
            mlp_param += len(s_param)
        elif 'attn' in name:
            attn_param += len(s_param)
        total_param += len(s_param)

    print('the num of parameters is {}'.format(total_param))
    print('the num of attention parameters is {}'.format(attn_param))

    return total_param, attn_param, mlp_param


if __name__ == '__main__':
    config = get_eval_config()
    config.checkpoint_path = '/home/maojunzhu/pycharmprojects/vision-transformer-pytorch/weights/imagenet21k+imagenet2012_ViT-B_32.pth'
    if config.checkpoint_path:
        state_dict, block_settings,_,_ = load_checkpoint(config.checkpoint_path)
        if block_settings is not None:
            config.block_settings = block_settings
    model = VisionTransformer(
            image_size=(config.image_size, config.image_size),
            patch_size=(config.patch_size, config.patch_size),
            emb_dim=config.emb_dim,
            mlp_dim=config.mlp_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            block_settings=config.block_settings,
            num_classes=config.num_classes,
            attn_dropout_rate=config.attn_dropout_rate,
            dropout_rate=config.dropout_rate)
    compute_param(model)
    print('The flops is: {:,}'.format(model.flops()))


