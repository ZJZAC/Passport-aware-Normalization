"""
Author: Benny
Date: Nov 2019
"""
import argparse
import numpy as np
import os
import torch

import sys
import importlib
import matplotlib.pyplot as plt



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))



'''PARAMETERS'''
parser = argparse.ArgumentParser('PointNet')
parser.add_argument('--seed', type=int, default=0, help=' seed value [default: 0]')
parser.add_argument('--batch_size', type=int, default=16, help='batch size in training [default: 24]')
parser.add_argument('--dataset', type=str, default="shapenet", help='Point Number [default: shapenet, modelnet]')
parser.add_argument('--model', default='pointnet_cls_ori_bn', help='model name [default: pointnet_cls_ori_bn, pointnet2_cls_ssg]')
parser.add_argument('--model2', default='pointnet_cls_all_bn', help='model name [default: pointnet_cls_ori_bn, pointnet2_cls_ssg]')
parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--num_class', type=int, default=16, help='Class Number [default: 40,16]')
parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
parser.add_argument('--epoch',  default=300, type=int, help='number of epoch in training [default: 300]')
parser.add_argument('--remark', type=str, default=None, help='exp remark')
parser.add_argument('--task', type=str, default='original', help='exp task')
parser.add_argument('--norm', type=str, default='BN', help='type of normlization [default: BN, BNCE, GN, GNCE]')
parser.add_argument('--task2', type=str, default='ALL', help='exp task')
parser.add_argument('--norm2', type=str, default='BNSE', help='type of normlization [default: BN, BNCE, GN, GNCE]')

def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)



def main():
    args = parser.parse_args()

    '''MODEL LOADING'''
    num_class = args.num_class
    MODEL = importlib.import_module(args.model)
    MODEL2 = importlib.import_module(args.model2)
    classifier = MODEL.get_model(num_class, channel=3)
    classifier2 = MODEL2.get_model(num_class, channel=3)
    pth_dir = '/data-x/g12/zhangjie/3dIP/week9-exp/v0/classification/' + args.dataset + "-" \
              + args.task + "-" + args.norm + "/checkpoints/best_model.pth"

    pth_dir2 = '/data-x/g12/zhangjie/3dIP/week9-exp/v2/classification/' + args.dataset + "-" \
              + args.task2 + "-" + args.norm2 + "/checkpoints/best_model.pth"

    checkpoint = torch.load(pth_dir)
    model_dict = checkpoint['model_state_dict']
    classifier.load_state_dict(model_dict)

    checkpoint2 = torch.load(pth_dir2)
    model_dict2 = checkpoint2['model_state_dict']
    classifier2.load_state_dict(model_dict2)

    p0 = classifier.conv3.weight
    p1 = classifier2.convp3.weight
    print(p0)
    print(p1)
    p0 = p0.flatten()
    p1 = p1.flatten()
    # x = np.arange(len(p))
    y0 = p0.detach().numpy()
    y1 = p1.detach().numpy()
    plt.figure()
    plt.hist(y0,density=True, bins=20, label="original")
    plt.hist(y1,density=True, bins=20, label="passport")
    plt.ylabel('Pcentage (%)')
    plt.xlabel('weights value')
    plt.legend(loc='upper right')
    savefig('weights.eps')



if __name__ == '__main__':
    main()

