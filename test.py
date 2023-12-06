import os
from config import cfg
import argparse
import timm
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger
from utils.load_check import load_check
from model.backbones.vit_pytorch import Attention

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--weight-path',
                        default='/home/maojunzhu/pycharmprojects/LA-Transformer/model/o-duke-ti16-attn40-1-0-6-lr5e-5-wd1e-4-kd-2-1-de03/net_best.pth')
    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    state_dict, channels, heads = load_check(cfg.TEST.WEIGHT)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num,
                       channels=channels, heads=heads)
    model.load_state_dict(state_dict, strict=False)
    if channels is not None:
        model.base.cal_seq()
    # for i in state_dict:
    #     model.state_dict()[i.replace('module.', '')].copy_(state_dict[i])
    print('Loading pretrained model for finetuning from {}'.format(cfg.TEST.WEIGHT))

    if heads is not None or channels is not None:
        del model.base.blocks[-1]
        model.base.local_feature = False

    n_parameters = sum(p.numel() for p in model.base.parameters() if p.requires_grad)
    msa_param = sum(p.numel() for n, p in model.base.named_parameters() if 'attn' in n)
    flops, attn_flops = model.base.flops()

    logger.info('number of params:{}'.format(n_parameters))
    logger.info('number of msa params:{}'.format(msa_param))
    logger.info('flops:{}'.format(flops))

    # for name, module in model.named_modules():
    #     if isinstance(module, Attention):
    #         print(module.disable_index)
    # assert 0

    # print(model)
    # checkpoint, channels, heads = load_check(args.weight_path)

    # channels = [149] * 12
    # print(checkpoint.keys())
    # heads = [int(dim/64) for dim in heads]
    # num_classes = checkpoint['model.head.weight'].shape[0]
    # num_classes = 702
    # print(num_classes)

    # vit_base = timm.create_model('amg_vit_tiny_patch16_224',
    #                              pretrained=False, channels=channels, num_heads=heads, num_classes=num_classes)
    # vit_base.load_state_dict(checkpoint, strict=False)
    # vit_base = vit_base.to(device)

    # print('flops:', vit_base.flops()[0], 'attn_flops:', vit_base.flops()[1])
    #
    # # Create La-Transformer
    # model = LATransformerTest(vit_base, lmbd=8)
    # model.load_state_dict(checkpoint, strict=False)

    if cfg.DATASETS.NAMES == 'VehicleID':
        for trial in range(10):
            train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
            rank_1, rank5 = do_inference(cfg,
                 model,
                 val_loader,
                 num_query)
            if trial == 0:
                all_rank_1 = rank_1
                all_rank_5 = rank5
            else:
                all_rank_1 = all_rank_1 + rank_1
                all_rank_5 = all_rank_5 + rank5

            logger.info("rank_1:{}, rank_5 {} : trial : {}".format(rank_1, rank5, trial))
        logger.info("sum_rank_1:{:.1%}, sum_rank_5 {:.1%}".format(all_rank_1.sum()/10.0, all_rank_5.sum()/10.0))
    else:
       do_inference(cfg,
                 model,
                 val_loader,
                 num_query)

