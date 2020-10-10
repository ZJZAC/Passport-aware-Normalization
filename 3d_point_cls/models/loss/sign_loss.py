import torch
import torch.nn as nn
import torch.nn.functional as F


class SignLoss(nn.Module):
    def __init__(self, alpha, b=None):
        super(SignLoss, self).__init__()
        self.alpha = alpha  #alpha 是网络结构中是否加sign loss的flag
        self.register_buffer('b', b) #将b注册到模型参数里
        self.loss = 0
        self.acc = 0
        self.scale_cache = None  #初始化


    def set_b(self, b):
        self.b.copy_(b)

    def get_acc(self):
        if self.scale_cache is not None:
            acc = (torch.sign(self.b.view(-1)) == torch.sign(self.scale_cache.view(-1))).float().mean()
            return acc
        else:
            raise Exception('scale_cache is None')  #一般不报错，因为现有scale，在计算signloss

    def get_loss(self):
        if self.scale_cache is not None:
            loss = (self.alpha * F.relu(-self.b.view(-1) * self.scale_cache.view(-1) + 0.1)).sum()  #view(-1)都展开类似成一位标量
            return loss
        else:
            raise Exception('scale_cache is None')

    def add(self, scale):
        self.scale_cache = scale
        self.loss += self.get_loss()
        self.loss += (0.00001 * scale.view(-1).pow(2).sum())  # to regularize the scale not to be so large,scale 的平方  正则项，限制scale不要太大
        self.acc += self.get_acc()

    def reset(self):
        self.loss = 0
        self.acc = 0
        self.scale_cache = None

