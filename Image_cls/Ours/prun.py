
import json
import os
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import prepare_dataset
from experiments.utils import construct_passport_kwargs_from_dict

from models.alexnet_passport_private import AlexNetPassportPrivate
from models.resnet_passport_private import ResNetPrivate
from models.layers.passportconv2d import PassportBlock
from models.layers.passportconv2d_private import PassportPrivateBlock

import  shutil
from experiments.logger import  Logger, savefig
import matplotlib.pyplot as plt

def test(model, device, dataloader, msg='Testing Result', ind=0):
    model.eval()
    device = device
    verbose = True
    loss_meter = 0
    acc_meter = 0
    runcount = 0

    start_time = time.time()
    with torch.no_grad():
        for load in dataloader:
            data, target = load[:2]
            data = data.to(device)
            target = target.to(device)

            pred = model(data, ind=ind)
            loss_meter += F.cross_entropy(pred, target, reduction='sum').item()  # sum up batch loss
            pred = pred.max(1, keepdim=True)[1]  # get the index of the max log-probability
            acc_meter += pred.eq(target.view_as(pred)).sum().item()
            runcount += data.size(0)

    loss_meter /= runcount
    acc_meter = 100 * acc_meter / runcount

    if verbose:
        print(f'{msg}: '
              f'Loss: {loss_meter:6.4f} '
              f'Acc: {acc_meter:6.2f} ({time.time() - start_time:.2f}s)')
        print()

    return {'loss': loss_meter, 'acc': acc_meter, 'time': time.time() - start_time}

def test_signature(model):
    model.eval()
    res = {}
    avg_private = 0
    avg_public = 0
    count_private = 0
    count_public = 0

    with torch.no_grad():
        for name, m in model.named_modules():
            if isinstance(m, PassportPrivateBlock):
                signbit, _ = m.get_scale(ind=1)
                signbit = signbit.view(-1).sign()
                privatebit = m.b

                detection = (signbit == privatebit).float().mean().item()
                res['private_' + name] = detection
                avg_private += detection
                count_private += 1

            if isinstance(m, PassportBlock):
                signbit = m.get_scale().view(-1).sign()
                publicbit = m.b

                detection = (signbit == publicbit).float().mean().item()
                res['public_' + name] = detection
                avg_public += detection
                count_public += 1

    pub_acc = 0
    pri_acc = 0
    if count_private != 0:
        print(f'Private Sign Detection Accuracy: {avg_private / count_private * 100:6.4f}')
        pri_acc = avg_private / count_private

    if count_public != 0:
        print(f'Public Sign Detection Accuracy: {avg_public / count_public * 100:6.4f}')


    return res,  pri_acc


def pruning_resnet(model, pruning_perc):
    if pruning_perc == 0:
        return

    allweights = []
    for p in model.parameters():
        allweights += p.data.cpu().abs().numpy().flatten().tolist()

    allweights = np.array(allweights)
    threshold = np.percentile(allweights, pruning_perc)

    for name,p in model.named_parameters():
        if  'fc' not in name :
            mask = p.abs() > threshold
            p.data.mul_(mask.float())

#
#
# device = torch.device('cuda')
# logdir = '/data-x/g12/zhangjie/DeepIPR/ours/resnet_cifar100_v3_all-our4/1'
# arch = 'resnet'
# passport_config = 'passport_configs/resnet18_passport.json'
# dataset  = 'cifar100'
#
# batch_size = 64
# nclass = 100 if dataset == 'cifar100' else 10
# inchan = 3
#
# print(logdir)
# prun_dir = logdir + '/prun'
# if not os.path.exists(prun_dir):
#     os.mkdir(prun_dir)
# shutil.copy('prun.py', str(prun_dir) + "/prun.py")
#
# title =''  #
# txt_pth = os.path.join(prun_dir, 'log_prun.txt')
# logger_prun = Logger(txt_pth, title=title)
# logger_prun.set_names([  'Model for Releasing  ', 'Model for Verification ', 'Signature'])
#
# trainloader, valloader = prepare_dataset({'transfer_learning': False,
#                                           'dataset': dataset,
#                                           'tl_dataset': '',
#                                           'batch_size': batch_size})
#
# passport_kwargs = construct_passport_kwargs_from_dict({'passport_config': json.load(open(passport_config)),
#                                                        'norm_type': 'bn',
#                                                        'sl_ratio': 0.1,
#                                                        'key_type': 'random'})
#
# if arch == 'alexnet':
#     model = AlexNetPassportPrivate(inchan, nclass, passport_kwargs)
# elif arch == 'resnet':
#     model = ResNetPrivate(inchan, nclass, passport_kwargs)
#
# sd = torch.load(logdir + '/models/best.pth')
# print(logdir + '/models/best.pth')
#
# model.load_state_dict(sd)
# model.to(device)
# pub_ori = test(model,device,valloader,msg='Ori_pub Result', ind=0)
# pri_ori = test(model,device,valloader,msg='Pri_pub Result', ind=1)
#
# for perc in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
#     sd = torch.load(logdir + '/models/best.pth')
#     model.load_state_dict(sd)
#     # pruning(model, PassportPrivateBlock, conf, -1, perc)
#     pruning_resnet(model, perc)
#     model.to(device)
#     res={}
#
#     res['perc'] =perc
#     res['pub_ori'] = test(model, device, valloader, msg='pruning %s percent Ori_pub Result'%perc, ind=0)
#     res['pri_ori']= test(model, device, valloader, msg='pruning %s percent Pri_pub Result'%perc, ind=1)
#     _,res['pri_sign_acc'] = test_signature(model)
#
#     pub_acc = res['pub_ori']['acc']
#     pri_acc = res['pri_ori']['acc']
#     pri_sign_acc = res['pri_sign_acc'] *100
#     logger_prun.append([pub_acc, pri_acc,pri_sign_acc])
#
# file = open(txt_pth, "r")
#
# name = file.readline()
# names = name.rstrip().split('\t')  # 标题分开
# numbers = {}
# for _, name in enumerate(names):
#     numbers[name] = []
#
# for line in file:
#     numbers_clm = line.rstrip().split('\t')
#
#     for clm in range(0, len(numbers_clm)):
#         numbers[names[clm]].append(numbers_clm[clm])
#
# #########plot####################
#
# plt.figure()
# names_legend = []
# for i, name in enumerate(names):
#
#     x = np.arange(len(numbers[name]))
#     num_float = []
#     for num in numbers[name]:
#         num_float.append(float(num))
#
#     names_legend.append(name)
#     x = np.arange(len(numbers[name])) * 10 #  start with 10%
#     plt.plot(x, num_float)
#
# plt.legend([name for name in names_legend], fontsize=10)
#
# plt.grid(True)
# plt.ylabel('Accuracy (%)', fontsize=10)
# plt.xlabel('Pruning Rate (%)', fontsize=10)
# save_name = prun_dir + '/prun.png'
# # plt.savefig(save_name)
# savefig(save_name)