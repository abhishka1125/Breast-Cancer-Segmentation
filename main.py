from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataset import TNBCDataset
from unet import TransUnet
from loss import DiceLoss
from eval_mat import BCEvaluator
from utils import create_logger, get_model_summary

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import argparse
import numpy as np
from tqdm import tqdm
import os
import pprint

def parse_args():
    parser = argparse.ArgumentParser(description='Train BC Segmentation')

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='weights')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='logs')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='dataset')
    parser.add_argument('--vit_blocks',
                        help='Number of ViT blocks',
                        type=int,
                        default=1)
    parser.add_argument('--img_dim',
                        help='input dim',
                        type=int,
                        default=512)
    parser.add_argument('--vit_dim_linear_mhsa_block',
                        help='Dimesntion of ViT Multi Head Self Attention blocks',
                        type=int,
                        default=512)
    parser.add_argument('--n_class',
                        help='Number of segmentation class',
                        type=int,
                        default=2)
    parser.add_argument('--train_batch',
                        help='Training Batch Size',
                        type=int,
                        default=4)
    parser.add_argument('--split_ratio',
                        help='train set valid set ratio',
                        type=float,
                        default=0.8)
    parser.add_argument('--epochs',
                        help='Total Epochs',
                        type=int,
                        default=100)
    parser.add_argument('--max_lr',
                        help='maximum Learning Rate',
                        type=float,
                        default=0.01)
    parser.add_argument('--weight_decay',
                        help='Weight Decay',
                        type=float,
                        default=1e-4)

    args = parser.parse_args()

    return args

def main():
    opts = parse_args()

    logger, tb_log_dir = create_logger(opts)
    logger.info(pprint.pformat(opts))

    # cudnn related setting
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    # define Model
    model = TransUnet(in_channels=3,
                        img_dim=opts.img_dim,
                        vit_blocks=opts.vit_blocks,
                        vit_dim_linear_mhsa_block=opts.vit_dim_linear_mhsa_block,
                        classes=opts.n_class)

    logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (2, 3, opts.img_dim, opts.img_dim)
    )
    writer_dict['writer'].add_graph(model, (dump_input, ))

    logger.info(get_model_summary(model, dump_input))

    model = model.cuda()


    # Preparing Dataset
    dataset = TNBCDataset(root=opts.dataDir)


    train_split = int(opts.split_ratio * len(dataset))
    val_split = len(dataset) - train_split
    torch.manual_seed(1)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_split, val_split])

    test_batch = 2

    train_loader = DataLoader(train_set, batch_size=opts.train_batch, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=test_batch, shuffle=True, pin_memory=True, num_workers=4)

    # Evaluation 
    evaluator = BCEvaluator()

    # Optim 
    optimizer = torch.optim.Adam(model.parameters(), opts.max_lr, weight_decay=opts.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, opts.max_lr, epochs=opts.epochs, steps_per_epoch=len(train_loader))

    ce_loss = torch.nn.CrossEntropyLoss()
    dice_loss = DiceLoss()

    # Training 

    val_dice = 1.0
    best_mIoU = 0
    best_fwIoU = 0
    best_mACC = 0
    best_pACC = 0

    for epoch in range(opts.epochs):
        model.train()
        losses = 0

        with tqdm(train_loader, unit="batch") as tepoch:
            for img, mask in tepoch:
                tepoch.set_description('Train Epoch {}'.format(epoch))

                img, mask = img.cuda(), mask.squeeze().long().cuda()

                optimizer.zero_grad()
                output = model(img.cuda())

                out_softmax = torch.argmax(torch.softmax(output, dim=1), dim=1)
                loss = 0.5 * ce_loss(output, mask) + 0.5 * dice_loss(out_softmax, mask)

                loss.backward()
                optimizer.step()
                sched.step()
                tepoch.set_postfix(loss=loss.item())

                losses += loss.item()
                
            train_loss = losses/len(train_loader)
        
        msg = 'Train Epoch : {}\t Training Loss : {}'.format(epoch, train_loss)
        logger.info(msg)

        global_steps = writer_dict['train_global_steps']
        writer_dict['writer'].add_scalar('train_loss', train_loss, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

        
        model.eval()
        losses = 0

        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as tepoch:
                for img, mask in tepoch:
                    tepoch.set_description('Train Epoch {}'.format(epoch))

                    img, mask = img.cuda(), mask.squeeze().long().cuda()

                    optimizer.zero_grad()
                    output = model(img.cuda())

                    out_softmax = torch.argmax(torch.softmax(output, dim=1), dim=1)
                    loss = 0.5 * ce_loss(output, mask) + 0.5 * dice_loss(out_softmax, mask)

                    tepoch.set_postfix(loss=loss.item())

                    losses += loss.item()

                    for i in range(test_batch):
                        evaluator.update(out_softmax[i].squeeze().cpu().numpy(), mask[i].squeeze().cpu().numpy())

        val_mats = evaluator.evaluate()

        val_loss = losses/len(val_loader)    

        # Logs
        msg = 'Validation Epoch : {}\t Validation Loss : {}\t Mean intersection-over-union averaged across classes : {}\t Frequency Weighted IoU : {}\t Mean pixel accuracy averaged across classes : {}\t Pixel Accuracy : {}'.format(
                        epoch, val_loss, val_mats['bin_seg']['mIoU'], val_mats['bin_seg']['fwIoU'], val_mats['bin_seg']['mACC'], val_mats['bin_seg']['pACC'])
        logger.info(msg)

        global_steps = writer_dict['valid_global_steps']
        writer_dict['writer'].add_scalar('val_loss', val_loss, global_steps)
        writer_dict['writer'].add_scalar('mIoU', val_mats['bin_seg']['mIoU'], global_steps)
        writer_dict['writer'].add_scalar('fwIoU', val_mats['bin_seg']['fwIoU'], global_steps)
        writer_dict['writer'].add_scalar('mACC', val_mats['bin_seg']['mACC'], global_steps)
        writer_dict['writer'].add_scalar('pACC', val_mats['bin_seg']['pACC'], global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1


        if val_loss < val_dice:

            logger.info('=> saving checkpoint to {}'.format(opts.modelDir))

            ckpt = {
                'model': model.state_dict(),
                'epoch': epoch,
                'loss': val_loss
            }

            ckpt.update(val_mats)

            torch.save(ckpt, os.path.join(opts.modelDir, 'best_model.pt'))
            
            val_dice = val_loss

        if val_mats['bin_seg']['mIoU'] > best_mIoU:
            best_mIoU = val_mats['bin_seg']['mIoU']

        if val_mats['bin_seg']['fwIoU'] > best_fwIoU:
            best_fwIoU = val_mats['bin_seg']['fwIoU']

        if val_mats['bin_seg']['mACC'] > best_mACC:
            best_mACC = val_mats['bin_seg']['mACC']

        if val_mats['bin_seg']['pACC'] > best_pACC:
            best_pACC = val_mats['bin_seg']['pACC']
        
    
    logger.info('Training Finished......')
    logger.info('Best Evaluation Matrics ...........')

    best_msg = 'Mean intersection-over-union averaged across classes : {}\t Frequency Weighted IoU : {}\t Mean pixel accuracy averaged across classes : {}\t Pixel Accuracy : {}'.format(
        best_mIoU, best_fwIoU, best_mACC, best_pACC)
    logger.info(best_msg)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
