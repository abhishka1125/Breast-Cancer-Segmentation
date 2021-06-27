from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from unet import TransUnet
from utils import outline

import torch
import torchvision.transforms.functional as TF

import cv2
import argparse
import numpy as np
import os
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='Infer BC Segmentation')

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='weights')
    parser.add_argument('--in_img',
                        help='input image',
                        type=str,
                        default='dataset/images/01_1.png')
    parser.add_argument('--gt_img',
                        help='input image',
                        type=str,
                        default='dataset/labels/01_1.png')
    parser.add_argument('--outDir',
                        help='Output directory',
                        type=str,
                        default='results')
    parser.add_argument('--vit_blocks',
                        help='Number of ViT blocks',
                        type=int,
                        default=4)
    parser.add_argument('--img_dim',
                        help='input dim',
                        type=int,
                        default=512)
    parser.add_argument('--vit_dim_linear_mhsa_block',
                        help='Dimesntion of ViT Multi Head Self Attention blocks',
                        type=int,
                        default=1024)
    parser.add_argument('--n_class',
                        help='Number of segmentation class',
                        type=int,
                        default=2)


    args = parser.parse_args()

    return args

def main():
    opts = parse_args()

    # define Model
    model = TransUnet(in_channels=3,
                        img_dim=opts.img_dim,
                        vit_blocks=opts.vit_blocks,
                        vit_dim_linear_mhsa_block=opts.vit_dim_linear_mhsa_block,
                        classes=opts.n_class)
    
    ckpt = torch.load(os.path.join(opts.modelDir, 'best_model.pt'))
    model.load_state_dict(ckpt['model'])
    model = model.cuda()
    model.eval()

    # Prepare input
    img = Image.open(opts.in_img).convert('RGB')
    img_np = np.array(img, dtype=np.uint8)
    img = TF.to_tensor(img)

    gt = Image.open(opts.gt_img).convert('L')
    gt_np = np.array(gt, dtype=np.uint8)

    with torch.no_grad():
        img = img.unsqueeze(0).repeat(2, 1, 1, 1)

        output = model(img.cuda())

        out_softmax = torch.argmax(torch.softmax(output, dim=1), dim=1)

        out_softmax_si = out_softmax[0]
        out_np = np.array(out_softmax_si.cpu().numpy() * 255, dtype=np.uint8)

    #out_np = outline(img_np, out_np, color=[255, 0, 0])
    pred_canny = cv2.Canny(out_np, 30, 150)
    gt_canny = cv2.Canny(gt_np, 30, 150)

    img_np[pred_canny == 255] = [255, 0, 0]  # turn edges to red
    img_np[gt_canny == 255] = [0, 255, 0]  # turn edges to red

    out_img = os.path.join(opts.outDir, opts.in_img.split('/')[-1])
    Image.fromarray(img_np).save(out_img)

    #print(out_softmax_si.shape, np.unique(out_np))

if __name__ == '__main__':
    main()