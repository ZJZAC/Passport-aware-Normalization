import json
import os
import time
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import prepare_dataset
from experiments.utils import construct_passport_kwargs_from_dict
from models.alexnet_passport_private import AlexNetPassportPrivate
from models.resnet_passport_private import ResNetPrivate
from models.layers.passportconv2d import PassportBlock
from models.layers.passportconv2d_private import PassportPrivateBlock
from models.losses.sign_loss import SignLoss
import shutil

class DatasetArgs():
    pass

def train_maximize(origpassport, fakepassport, model, optimizer, criterion, trainloader, device, type):
    model.train()
    loss_meter = 0
    signloss_meter = 0
    maximizeloss_meter = 0
    mseloss_meter = 0
    csloss_meter = 0
    acc_meter = 0
    signacc_meter = 0
    start_time = time.time()
    mse_criterion = nn.MSELoss()
    cs_criterion = nn.CosineSimilarity()   #余弦相似性
    for k, (d, t) in enumerate(trainloader):
        d = d.to(device)
        t = t.to(device)

        optimizer.zero_grad()

        # if scheme == 1:
        #     pred = model(d)
        # else:
        pred = model(d, ind=1)  #private graph

        loss = criterion(pred, t)

        signloss = torch.tensor(0.).to(device)
        signacc = torch.tensor(0.).to(device)
        count = 0
        for m in model.modules():
            if isinstance(m, SignLoss):
                signloss += m.loss
                signacc += m.acc
                count += 1

        maximizeloss = torch.tensor(0.).to(device)
        mseloss = torch.tensor(0.).to(device)
        csloss = torch.tensor(0.).to(device)
        for l, r in zip(origpassport, fakepassport):
            mse = mse_criterion(l, r)
            cs = cs_criterion(l.view(1, -1), r.view(1, -1)).mean()
            csloss += cs
            mseloss += mse
            maximizeloss += 1 / mse

        if 'fake2-' in type  :
            (loss).backward()  #only cross-entropy loss  backward  fake2
        elif  'fake3-' in type :
            (loss + maximizeloss).backward()  #csloss do not backward   kafe3

        else:
            # print("FFFFFFFFFFFFFFFFFF")
            (loss + maximizeloss + signloss).backward()  #csloss  backward   #fake3_S

        torch.nn.utils.clip_grad_norm_(fakepassport, 2)  #梯度裁剪

        optimizer.step()

        acc = (pred.max(dim=1)[1] == t).float().mean()

        loss_meter += loss.item()
        acc_meter += acc.item()
        signloss_meter += signloss.item()
        signacc_meter += signacc.item() / count
        maximizeloss_meter += maximizeloss.item()
        mseloss_meter += mseloss.item()
        csloss_meter += csloss.item()

        print(f'Batch [{k + 1}/{len(trainloader)}]: '
              f'Loss: {loss_meter / (k + 1):.4f} '
              f'Acc: {acc_meter / (k + 1):.4f} '
              f'Sign Loss: {signloss_meter / (k + 1):.4f} '
              f'Sign Acc: {signacc_meter / (k + 1):.4f} '
              f'MSE Loss: {mseloss_meter / (k + 1):.4f} '
              f'Maximize Dist: {maximizeloss_meter / (k + 1):.4f} '
              f'CS: {csloss_meter / (k + 1):.4f} ({time.time() - start_time:.2f}s)',
              end='\r')

    print()
    loss_meter /= len(trainloader)
    acc_meter /= len(trainloader)
    signloss_meter /= len(trainloader)
    signacc_meter /= len(trainloader)
    maximizeloss_meter /= len(trainloader)
    mseloss_meter /= len(trainloader)
    csloss_meter /= len(trainloader)

    return {'loss': loss_meter,
            'signloss': signloss_meter,
            'acc': acc_meter,
            'signacc': signacc_meter,
            'maximizeloss': maximizeloss_meter,
            'mseloss': mseloss_meter,
            'csloss': csloss_meter,
            'time': start_time - time.time()}


def test_fake(model, criterion, valloader, device):
    model.eval()
    loss_meter = 0
    signloss_meter = 0
    acc_meter = 0
    signacc_meter = 0
    start_time = time.time()

    with torch.no_grad():
        for k, (d, t) in enumerate(valloader):
            d = d.to(device)
            t = t.to(device)

            # if scheme == 1:
            #     pred = model(d)
            # else:
            pred = model(d, ind=1)

            loss = criterion(pred, t)

            signloss = torch.tensor(0.).to(device)
            signacc = torch.tensor(0.).to(device)
            count = 0

            for m in model.modules():
                if isinstance(m, SignLoss):
                    signloss += m.get_loss()
                    signacc += m.get_acc()
                    count += 1

            acc = (pred.max(dim=1)[1] == t).float().mean()

            loss_meter += loss.item()
            acc_meter += acc.item()
            signloss_meter += signloss.item()
            try:
                signacc_meter += signacc.item() / count
            except:
                pass

            print(f'Batch [{k + 1}/{len(valloader)}]: '
                  f'Loss: {loss_meter / (k + 1):.4f} '
                  f'Acc: {acc_meter / (k + 1):.4f} '
                  f'Sign Loss: {signloss_meter / (k + 1):.4f} '
                  f'Sign Acc: {signacc_meter / (k + 1):.4f} ({time.time() - start_time:.2f}s)',
                  end='\r')

    print()

    loss_meter /= len(valloader)
    acc_meter /= len(valloader)
    signloss_meter /= len(valloader)
    signacc_meter /= len(valloader)

    return {'loss': loss_meter,
            'signloss': signloss_meter,
            'acc': acc_meter,
            'signacc': signacc_meter,
            'time': time.time() - start_time}




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='fake attack 3: create another passport maximized from current passport')
    # parser.add_argument('--rep', default=1, type=int,  help='training id')
    parser.add_argument('--rep', default=1, type=str,  help='training comment')
    parser.add_argument('--arch', default='resnet18', choices=['alexnet', 'resnet18'])
    parser.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'])
    parser.add_argument('--flipperc', default=0, type=float,
                        help='flip percentange 0~1')
    parser.add_argument('--scheme', default=1, choices=[1, 2, 3], type=int)
    parser.add_argument('--loadpath', default='', help='path to model to be attacked')
    parser.add_argument('--passport-config', default='', help='path to passport config')
    args = parser.parse_args()

    args.scheme = 3
    args.loadpath = "/data-x/g12/zhangjie/DeepIPR/baseline/resnet18_cifar100_v3_all/"
    # print(args.loadpath.split('/')[-2])
    # sys.exit(0)
    print(args.loadpath)
