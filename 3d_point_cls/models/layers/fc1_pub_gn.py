import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


class fc1_pub_gn(nn.Module):
    def __init__(self, i, o):
        super().__init__()

        self.linear = nn.Linear(i, o)
        self.weight = self.linear.weight
        self.gn = nn.GroupNorm(o // 16, o, affine=True)



    def forward(self, x, ind=0):

        x = self.conv(x)
        x = self.gn(x)

        return x
