import numpy as np

import torch
import pickle
import torch.nn as nn
from model import VisionTransformer, SelfAttention, LinearGeneral


def normalize_ranks_per_layer(layer_ranks):
    for i in range(len(layer_ranks)):
        v = torch.abs(layer_ranks[i])
        v = v / torch.sqrt(torch.sum(v * v))
        layer_ranks[i] = v
    return layer_ranks


def get_new_out(module, idx):
    heads = len(idx)
    head_dim = module.weight.data.size(1)
    in_dim = module.weight.data.size(2)
    new_out = LinearGeneral((heads, head_dim), (in_dim,))
    w = module.weight.data[idx, :, :].clone()
    new_out.weight.data = w.clone()
    new_out.bias.data = module.bias.data.clone()
    return new_out


def get_new_attn(module, idx):
    in_dim = module.weight.data.size(0)
    head_dim = module.weight.data.size(2)
    heads = len(idx)
    new_attn = LinearGeneral((in_dim,), (heads, head_dim))
    w = module.weight.data[:, idx, :].clone()
    new_attn.weight.data = w.clone()
    b = module.bias.data[idx, :].clone()
    new_attn.bias.data = b.clone()
    return new_attn


def get_new_conv(model, idx):
    num_seq = len(idx)
    new_conv = nn.Conv2d(num_seq, num_seq, kernel_size=1, stride=1, bias=None, groups=num_seq)
    new_weight = model.weight.data[idx, :, :, :].clone()
    new_conv.weight.data = new_weight.clone()
    return new_conv


def prune_token(model, masks):
    model.cpu()
    layer = 0
    for name, module in model.named_modules():
        if isinstance(module, SelfAttention):
            idx = np.squeeze(np.argwhere(np.asarray(masks[layer])))

            module.index = nn.Parameter(torch.tensor(idx+1), requires_grad=False)

            module.reset_rank()
            layer += 1


def prune_head(model, masks):
    model.cpu()
    layer = 0
    for name, module in model.named_modules():
        if isinstance(module, SelfAttention):
            idx = np.squeeze(np.argwhere(np.asarray(masks[layer])))

            new_query = get_new_attn(module.query, idx)
            new_key = get_new_attn(module.key, idx)
            new_value = get_new_attn(module.value, idx)
            new_out = get_new_out(module.out, idx)

            module.query = new_query
            module.key = new_key
            module.value = new_value
            module.out = new_out

            module.reset_rank()

            layer += 1


def prune(model, nums, mode='head', learned_maps=None):
    layer_ranks = []
    attn_maps = []
    iter_cnts = []
    if mode == 'head':
        for name, module in model.named_modules():
            if isinstance(module, SelfAttention):
                layer_ranks.append(module.head_ranks)

    elif mode == 'seq':
        for name, module in model.named_modules():
            if isinstance(module, SelfAttention):
                layer_ranks.append(module.seq_ranks)
    elif mode == 'random':
        for name, module in model.named_modules():
            if isinstance(module, SelfAttention):
                num = module.index.data.shape[0]
                layer_ranks.append(torch.rand(num))
    elif mode == 'hrand':
        for name, module in model.named_modules():
            if isinstance(module, SelfAttention):
                attn_maps.append(torch.rand(module.query.weight.data.shape[1]))
                # print(module.query.weight.data.shape[1])
    elif mode == 'head_entropy':
        if learned_maps is None:
            for name, module in model.named_modules():
                if isinstance(module, SelfAttention):
                    attn_maps.append(module.attn.to('cpu'))
                    iter_cnts.append(module.cnt)

    else:
        print('no such mode')
        assert 0
    normalize_ranks_per_layer(layer_ranks)
    # print(layer_ranks)
    layer_ranks = np.asarray([np.asarray(layer_rank.cpu()) for layer_rank in layer_ranks])
    if mode == 'random':
        layers = len(layer_ranks)
        for layer in np.random.randint(layers, size=nums):
            length = len(layer_ranks[layer])
            id = np.random.randint(length)
            layer_ranks[layer][id] = 0
        masks = np.asarray([layer_rank != 0 for layer_rank in layer_ranks])
    # print(np.hstack(layer_ranks))
    elif mode == 'head_entropy':
        # if learned_maps is not None:
        #     # print('input head maps:' + learned_maps)
        #     heads_entropy = learned_maps
        heads_entropy = []
        for attn_map, cnt in zip(attn_maps, iter_cnts):
            b, h, Nq, Nk = attn_map.shape
            attn_map = (attn_map.sum(dim=0) / (b * cnt)).data
            # print(attn_map.shape)
            entropy_map = torch.zeros(attn_map.shape)
            for ki in range(Nk):
                entropy_map[:, :, ki] = -attn_map[:, :, ki] * torch.log2(attn_map[:, :, ki])
            head_entropy = entropy_map.sum(dim=2).mean(dim=1)
            heads_entropy.append(np.asarray(head_entropy))
            # print('heads entropy:' + heads_entropy)
        cut_layers = []
        for head_entropy in heads_entropy:
            if len(head_entropy) > 2:
                cut_layers.append(head_entropy)
        smallest = np.sort(np.hstack(cut_layers))[-nums]
        masks = []
        for head_entropy in heads_entropy:
            if len(head_entropy) <= 2:
                masks.append(head_entropy == head_entropy)
            else:
                masks.append(head_entropy < smallest)
        # print(masks)
    elif mode == 'hrand':
        smallest = np.sort(np.hstack(attn_maps))[-nums]
        masks = []
        for attn_map in attn_maps:
            if len(attn_map) < 2:
                masks.append(attn_map == attn_map)
            else:
                masks.append(attn_map < smallest)
    else:
        smallest = np.sort(np.hstack(layer_ranks))[nums]
        # print(smallest)
        masks = np.asarray([layer_rank >= smallest for layer_rank in layer_ranks])

    if mode == 'head' or mode == 'head_entropy' or mode == 'hrand':
        prune_head(model, masks)

        head_sets = [mask.sum() for mask in masks]
        head_sets = [head_set * 64 for head_set in head_sets]
        print(head_sets)
        return head_sets

    elif mode == 'seq' or mode == 'random':
        prune_token(model, masks)

        channels = [mask.sum() for mask in masks]
        print(channels)
        return channels

