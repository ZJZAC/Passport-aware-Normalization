import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


class fc1_pub_bn(nn.Module):
    def __init__(self, i, o):
        super().__init__()

        self.linear = nn.Linear(i, o)
        self.weight = self.linear.weight
        self.bn0 = nn.BatchNorm1d(o)
        self.bn1 = nn.BatchNorm1d(o)


    def forward(self, x, ind=0):

        x = self.linear(x)

        if ind == 0:
            x = self.bn0(x)
        else:
            x = self.bn1(x)

        return x
