import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


class fc2_pub_bn(nn.Module):
    def __init__(self, i, o):
        super().__init__()

        self.linear = nn.Linear(i, o, )
        self.weight = self.linear.weight
        self.gn = nn.GroupNorm(o // 16, o, affine=True)
        self.dropout = nn.Dropout(p=0.4)


    def forward(self, x,  ind=0):

        x = self.linear(x)
        x = self.dropout(x)

        x = self.conv(x)
        x = self.gn(x)

        return x
