import torch
import os
import numpy as np

from torch import nn
from model import VisionTransformer, SelfAttention
from config import get_prune_config
from checkpoint import load_checkpoint
from data_loaders import *
from utils import setup_device, accuracy, MetricTracker, TensorboardWriter, process_config, log_attns

from train import train_epoch, valid_epoch
from compute_params import compute_param
from prune import prune


def save_model(save_dir, epoch, model, optimizer, device_ids,
               block_settings=[768]*12, iter=None, channels=None, hidden_set=None):
    if block_settings is None:
        block_settings = [768]*12
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict() if len(device_ids) <= 1 else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'block_settings': block_settings,
        'hidden_set': hidden_set,
        'channels': channels
    }
    if iter is not None:
        filename = str(save_dir + 'iter{}.pth'.format(iter))
    else:
        filename = str(save_dir + 'finetune{}.pth'.format(epoch))
    torch.save(state, filename)


def train_epoch_prune(iter, model, data_loader, criterion, optimizer, metrics,
                      mode='head', lr_scheduler=None, device=torch.device('cpu')):
    metrics.reset()
    model.to(device)
    # training loop
    for batch_idx, (batch_data, batch_target) in enumerate(data_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        optimizer.zero_grad()
        if mode == 'head':
            batch_pred = model(batch_data, compute_taylor_h=True)
        elif mode == 'seq':
            batch_pred = model(batch_data, compute_taylor_n=True)
        elif mode == 'attn':
            batch_pred = model(batch_data, compute_taylor_attn=True)
        elif mode == 'head_entropy':
            batch_pred = model(batch_data, compute_entropy=True)
        else:
            batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        acc1, acc5 = accuracy(batch_pred, batch_target, topk=(1, 5))
        # metrics.writer.set_step((epoch - 1) * len(data_loader) + batch_idx)
        metrics.update('loss', loss.item())
        metrics.update('acc1', acc1.item())
        metrics.update('acc5', acc5.item())

        if batch_idx % 10 == 0:
            print("prune iter: {:03d} Batch: {:05d}/{:05d} Loss: {:.4f} Acc@1: {:.2f}, Acc@5: {:.2f}"
                  .format(iter, batch_idx, len(data_loader), loss.item(), acc1.item(), acc5.item()))
            break
    return metrics.result()


def main(config):
    device, device_ids = setup_device(config.n_gpu)
    # set metrics
    metric_names = ['loss', 'acc1', 'acc5']
    train_metrics = MetricTracker(*[metric for metric in metric_names], writer=None)
    valid_metrics = MetricTracker(*[metric for metric in metric_names], writer=None)

    print("create dataloaders")
    train_dataloader = eval("{}DataLoader".format(config.dataset))(
        data_dir=os.path.join(config.data_dir, config.dataset),
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        split='train')

    valid_dataloader = eval("{}DataLoader".format(config.dataset))(
        data_dir=os.path.join(config.data_dir, config.dataset),
        image_size=config.image_size,
        batch_size=config.batch_size * 2,
        num_workers=config.num_workers,
        split='val')
    print('creat model')

    if config.checkpoint_path:

        state_dict, block_settings, channels = load_checkpoint(config.checkpoint_path)
        if block_settings is not None:
            config.block_settings = block_settings

        print('block_settings:{}'.format(config.block_settings))

        print('channels:{}'.format(channels))
    else:

        print('there must be a pre-trained model!')
        assert 0

    model = VisionTransformer(
        image_size=(config.image_size, config.image_size),
        patch_size=(config.patch_size, config.patch_size),
        emb_dim=config.emb_dim,
        mlp_dim=config.mlp_dim,
        block_settings=config.block_settings,
        channels=channels,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        attn_dropout_rate=config.attn_dropout_rate,
        dropout_rate=config.dropout_rate)

    print("Load pretrained weights from {}".format(config.checkpoint_path))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    logfile = open('./logs/' + config.exp_name + '.txt', 'w')

    if config.prune_mode == 'head' or config.prune_mode == 'head_entropy' \
       or config.prune_mode== 'hrand':
        num_to_prune = int(config.num_heads * config.num_layers * config.prune_rate)
        num_per_iter = int(num_to_prune / (config.iter_nums - 1e-6))
    elif config.prune_mode == 'seq' or config.prune_mode == 'attn' or config.prune_mode == 'random':
        h, w = config.image_size, config.image_size
        fh, fw = config.patch_size, config.patch_size
        gh, gw = h//fh, w//fw
        num_patches = gh * gw
        num_per_iter = int(num_patches * config.num_layers * config.prune_rate / config.iter_nums)

    prune_mode = config.prune_mode

    for it in range(config.iter_nums):
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=config.lr,
            weight_decay=config.wd,
            momentum=0.9)
        model.train()

        result = train_epoch_prune(it, model, train_dataloader, criterion,
                                   optimizer, train_metrics, mode=prune_mode, device=device)

        if config.prune_mode == 'head':
            block_settings = prune(model, num_per_iter, mode='head')
        elif config.prune_mode == 'seq' or config.prune_mode == 'attn':
            channels = prune(model, num_per_iter, mode='seq')
        elif config.prune_mode == 'random':
            channels = prune(model, num_per_iter, mode='random')
        elif config.prune_mode == 'hrand':
            block_settings = prune(model, num_per_iter, mode='hrand')
        elif config.prune_mode == 'head_entropy':
            block_settings = prune(model, num_per_iter, mode='head_entropy',)

        params, attn_params, _ = compute_param(model)

        model.to(device)
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=config.lr,
            weight_decay=config.wd,
            momentum=0.9)

        epoch = 0
        for epoch in range(1, config.finetune_nums+1):
            log = {'epoch': epoch}

            result = train_epoch(epoch, model, train_dataloader, criterion,
                                 optimizer, train_metrics, device=device)
            log.update(result)

            # validate the model
            model.eval()
            result = valid_epoch(epoch, model, valid_dataloader, criterion, valid_metrics, device)
            # log result
            logfile.writelines('iter:{} epoch:{} params:{} attn_params:{} acc:'
                               .format(it, epoch, params, attn_params) + str(result['acc1']) + '\n')
            logfile.flush()
            log.update(**{'val_' + k: v for k, v in result.items()})
            for key, value in log.items():
                print('    {:15s}: {}'.format(str(key), value))
        print(block_settings)
        save_model(config.checkpoint_dir, epoch, model, optimizer,
                   device_ids, block_settings=block_settings, channels=channels, iter=it)

    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=config.lr,
        weight_decay=config.wd,
        momentum=0.9)

    for epoch in range(config.final_finetune):
        log = {'epoch': epoch}

        result = train_epoch(epoch, model, train_dataloader, criterion,
                             optimizer, train_metrics, device=device)
        params, attn_params, _ = compute_param(model)
        log.update(result)

        # validate the model
        model.eval()
        result = valid_epoch(epoch, model, valid_dataloader, criterion, valid_metrics, device)
        # log result
        logfile.writelines('finetune epoch:{} params:{} attn_params:{} acc:'
                           .format(epoch, params, attn_params) + str(result['acc1']) + '\n')
        logfile.flush()
        log.update(**{'val_' + k: v for k, v in result.items()})
        for key, value in log.items():
            print('    {:15s}: {}'.format(str(key), value))

        save_model(config.checkpoint_dir, epoch, model, optimizer,
                   device_ids, block_settings=block_settings, channels=channels)


if __name__ == '__main__':
    cfg = get_prune_config()
    main(cfg)
