from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss, DistillationLoss
from processor import do_train, train_one_epoch
import random
import logging
import torch
import numpy as np
import os
import argparse
import copy

from config import cfg
from utils.metrics import R1_mAP_eval
from utils.load_check import load_check
from torch.cuda import amp
from prune import prune
import torch.distributed as dist

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID TransReid pruning")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    state_dict, channels, heads = load_check(cfg.FINETUNE.PRETRAIN_PATH)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num,
                       channels=channels, heads=heads)
    model.load_state_dict(state_dict, strict=None)
    if channels is not None:
        model.base.cal_seq()
    # model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    #
    # model.load_param(cfg.FINETUNE.PRETRAIN_PATH)

    teacher_model = None
    if cfg.FINETUNE.DISTILL_TYPE != 'none':
        teacher_model = copy.deepcopy(model)
        teacher_model.to(args.local_rank)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    distill_loss = DistillationLoss(loss_func, teacher_model, cfg.FINETUNE.DISTILL_TYPE,
                                    cfg.FINETUNE.DISTILL_ALPHA, cfg.FINETUNE.DISTILL_TAU)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)

    n_parameters = sum(p.numel() for p in model.base.parameters())
    print('number of params:', n_parameters)

    device = 'cuda'

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(args.local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              find_unused_parameters=True)

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    compute_attn = False
    compute_entro = False

    depth = len(model.base.blocks) - 1
    num_heads = model.base.blocks[0].attn.num_heads
    if cfg.FINETUNE.PRUNE_MODE == 'head_entropy':
        compute_entro = True
        num_to_prune = int(num_heads * depth * cfg.FINETUNE.PRUNE_RATE)
        print('num to prune:', num_to_prune)
        num_per_iter = int(num_to_prune / cfg.FINETUNE.ITER_NUM)

        # del model.base.blocks[-1]

    elif cfg.FINETUNE.PRUNE_MODE == 'attn':
        compute_attn = True
        num_patches = model.base.patch_embed.num_patches

        num_to_prune = int(num_patches * depth * cfg.FINETUNE.PRUNE_RATE)
        print('num to prune:', num_to_prune)
        num_per_iter = int(num_to_prune / cfg.FINETUNE.ITER_NUM)

    dims = None

    for iter in range(cfg.FINETUNE.ITER_NUM):

        train_one_epoch(cfg, model, iter, 0, evaluator, distill_loss, scaler, train_loader, val_loader, logger,
                        center_criterion, mode=cfg.FINETUNE.PRUNE_MODE, scheduler=scheduler, optimizer=None,
                        optimizer_center=None)

        if cfg.FINETUNE.PRUNE_MODE == 'head_entropy':
            heads, dims = prune(model.base, num_per_iter, mode='head_entropy')
        elif cfg.FINETUNE.PRUNE_MODE == 'attn':
            # model.base.blocks[-1].attn.disable_index = False
            channels = prune(model.base, num_per_iter, mode='attn')
        else:
            print('Prune nothing')

        model.to(args.local_rank)

        optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

        for epoch in range(1, cfg.FINETUNE.FINE_NUM + 1):

            train_one_epoch(cfg, model, iter, epoch, evaluator, distill_loss, scaler, train_loader, val_loader, logger,
                            center_criterion, mode=None, scheduler=scheduler, optimizer=optimizer,
                            optimizer_center=optimizer_center)

            save_dict = {
                'state_dict': model.state_dict(),
                'channels': channels,
                'heads': heads,
                'dims': dims,
            }
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(save_dict,
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_iter.pth'))
            else:
                torch.save(save_dict,
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_iter.pth'))

    best = 0
    for epoch in range(cfg.FINETUNE.FINAL_NUM):
        mAP = train_one_epoch(cfg, model, -1, epoch, evaluator, distill_loss, scaler, train_loader, val_loader, logger,
                        center_criterion, mode=None, scheduler=scheduler, optimizer=optimizer,
                        optimizer_center=optimizer_center)

        if mAP > best:
            is_best = True
            best = mAP
        else:
            is_best = False
        save_dict = {
            'state_dict': model.state_dict(),
            'channels': channels,
            'heads': heads,
            'dims': dims,
        }
        if cfg.MODEL.DIST_TRAIN:
            if dist.get_rank() == 0:
                if is_best:
                    torch.save(save_dict,
                               os.path.join(cfg.OUTPUT_DIR, 'best.pth'))
                else:
                    torch.save(save_dict,
                               os.path.join(cfg.OUTPUT_DIR, 'checkpoint.pth'))
        else:
            if is_best:
                torch.save(save_dict,
                           os.path.join(cfg.OUTPUT_DIR, 'best.pth'))
            else:
                torch.save(save_dict,
                           os.path.join(cfg.OUTPUT_DIR, 'checkpoint.pth'))
