import argparse
import logging
import os
import pprint
import random

import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from dataset.hypersim import Hypersim
from dataset.kitti import KITTI
from dataset.vkitti2 import VKITTI2
from dataset.soccernet import SoccerNet
from depth_anything_v2.dpt import DepthAnythingV2
from util.dist_helper import setup_distributed
from util.loss import SiLogLoss
from util.metric import eval_depth
from util.utils import init_log

parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')

parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--dataset', default='soccernet', choices=['hypersim', 'vkitti', 'soccernet'])
parser.add_argument('--img-size', default=518, type=int)
parser.add_argument('--min-depth', default=0.001, type=float)
parser.add_argument('--max-depth', default=250, type=float)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--bs', default=2, type=int)
parser.add_argument('--lr', default=0.000001, type=float)
parser.add_argument('--pretrained-from', type=str,default='/home/ipad_3d/dmy/Depth-Anything-V2/checkpoints/depth_anything_v2_${encoder}.pth')
parser.add_argument('--save-path', type=str, default='/data/ipad_3d/dmy/depthanything_output/logs/0')


def main():
    args = parser.parse_args()

    args.pretrained_from = f'/home/ipad_3d/dmy/Depth-Anything-V2/checkpoints/depth_anything_v2_{args.encoder}.pth'

    warnings.simplefilter('ignore', np.RankWarning)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter(args.save_path)

    cudnn.enabled = True
    cudnn.benchmark = True

    size = (args.img_size, args.img_size)
    if args.dataset == 'hypersim':
        trainset = Hypersim('dataset/splits/hypersim/train.txt', 'train', size=size)
    elif args.dataset == 'vkitti':
        trainset = VKITTI2('/home/ipad_3d/dmy/Marigold/data_split/vkitti/vkitti_train.txt', 'train', size=size)
    elif args.dataset == 'soccernet':
        trainset = SoccerNet('/public_datasets/SoccerNet/football/train/Train', 'train')
    else:
        raise NotImplementedError

    trainloader = DataLoader(
        trainset,
        batch_size=args.bs,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
    )

    if args.dataset == 'hypersim':
        valset = Hypersim('dataset/splits/hypersim/val.txt', 'val', size=size)
    elif args.dataset == 'vkitti':
        valset = KITTI('dataset/splits/kitti/val.txt', 'val', size=size)
    elif args.dataset == 'soccernet':
        valset = SoccerNet('/public_datasets/SoccerNet/football/val/Validation', 'val')
    else:
        raise NotImplementedError

    valloader = DataLoader(
        valset,
        batch_size=1,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
    )


    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})

    if args.pretrained_from:
        model.load_state_dict(
            {k: v for k, v in torch.load(args.pretrained_from, map_location='cpu').items() if 'pretrained' in k},
            strict=False)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)

    criterion = nn.MSELoss().to(device)#SiLogLoss().to(device)

    optimizer = AdamW(
        [{'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr},
         {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name],
          'lr': args.lr * 10.0}],
        lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    total_iters = args.epochs * len(trainloader)

    previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100,
                     'log10': 100, 'silog': 100}

    for epoch in range(args.epochs):

        logger.info('===========> Epoch: {:}/{:}, d1: {:.3f}, d2: {:.3f}, d3: {:.3f}'.format(epoch, args.epochs,
                                                                                                 previous_best['d1'],
                                                                                                 previous_best['d2'],
                                                                                                 previous_best['d3']))
        logger.info('===========> Epoch: {:}/{:}, abs_rel: {:.3f}, sq_rel: {:.3f}, rmse: {:.3f}, rmse_log: {:.3f}, '
                        'log10: {:.3f}, silog: {:.3f}'.format(
                epoch, args.epochs, previous_best['abs_rel'], previous_best['sq_rel'], previous_best['rmse'],
                previous_best['rmse_log'], previous_best['log10'], previous_best['silog']))

        #trainloader.sampler.set_epoch(epoch + 1)

        model.train()
        total_loss = 0

        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()

            img, depth, valid_mask = sample['image'].to(device), sample['depth'].to(device), sample['valid_mask'].to(device)

            if random.random() < 0.5:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)

            pred = model(img)
            pred = torch.stack([pred]*3,dim=-1)

            """
            with torch.no_grad():
                pred_temp = pred.detach().cpu().numpy()
                pred_temp = 255*(pred_temp[0] - pred_temp[0].min()) / (pred_temp[0].max() - pred_temp[0].min())
                pred_temp = pred_temp.astype(np.uint8)
                depth_temp = depth.detach().cpu().numpy()
                depth_temp = 255 * (depth_temp[0] - depth_temp[0].min()) / (depth_temp[0].max() - depth_temp[0].min())
                mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
                mask = mask.detach().cpu().numpy()[0]
                #mask = mask.astype(np.uint8) * 255
                data = np.concatenate([pred_temp[mask], depth_temp[mask]], axis=-1)
                plt.imshow(data,cmap='gray')
                plt.show()
            """

            mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
            loss = criterion(pred[mask], depth[mask])

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            iters = epoch * len(trainloader) + i

            lr = args.lr * (1 - iters / total_iters) ** 0.9

            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10.0

            writer.add_scalar('train/loss', loss.item(), iters)

            if i % 100 == 0:
                logger.info(
                    'Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(trainloader), optimizer.param_groups[0]['lr'],
                                                                   loss.item()))

        model.eval()

        results = {'d1': torch.tensor([0.0]).to(device), 'd2': torch.tensor([0.0]).to(device), 'd3': torch.tensor([0.0]).to(device),
                   'abs_rel': torch.tensor([0.0]).to(device), 'sq_rel': torch.tensor([0.0]).to(device),
                   'rmse': torch.tensor([0.0]).to(device),
                   'rmse_log': torch.tensor([0.0]).to(device), 'log10': torch.tensor([0.0]).to(device),
                   'silog': torch.tensor([0.0]).to(device)}
        nsamples = torch.tensor([0.0]).to(device)

        for i, sample in enumerate(valloader):

            img, depth, valid_mask = sample['image'].to(device).float(), sample['depth'].to(device)[0], \
            sample['valid_mask'].to(device)[0]

            with torch.no_grad():
                pred = model(img)
                pred = F.interpolate(pred[:, None], depth.shape, mode='bilinear', align_corners=True)[0, 0]

            valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)

            if valid_mask.sum() < 10:
                continue

            cur_results = eval_depth(pred[valid_mask], depth[valid_mask])

            for k in results.keys():
                results[k] += cur_results[k]
            nsamples += 1

        torch.distributed.barrier()

        for k in results.keys():
            dist.reduce(results[k], dst=0)
        dist.reduce(nsamples, dst=0)


        logger.info('==========================================================================================')
        logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
        logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(
                *tuple([(v / nsamples).item() for v in results.values()])))
        logger.info('==========================================================================================')


        for name, metric in results.items():
            writer.add_scalar(f'eval/{name}', (metric / nsamples).item(), epoch)

        for k in results.keys():
            if k in ['d1', 'd2', 'd3']:
                previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
            else:
                previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())


        checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
        torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))


if __name__ == '__main__':
    main()