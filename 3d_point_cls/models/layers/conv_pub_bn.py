import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


class conv_pub_bn(nn.Module):
    def __init__(self, i, o, ks=1):
        super().__init__()

        self.conv = nn.Conv1d(i, o, ks)
        self.bn0 = nn.BatchNorm1d(o)
        self.bn1 = nn.BatchNorm1d(o)
        self.weight = self.conv.weight

    def forward(self, x,  ind=0):

        x = self.conv(x)

        if ind ==0:
            x = self.bn0(x)
        else:
            x = self.bn1(x)

        return x
