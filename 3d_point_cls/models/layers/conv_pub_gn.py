import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


class conv_pub_gn(nn.Module):
    def __init__(self, i, o, ks=1):
        super().__init__()

        self.conv = nn.Conv1d(i, o, ks)
        self.gn = nn.GroupNorm(o // 16, o, affine=True)
        self.weight = self.conv.weight

    def forward(self, x,  ind=0):

        x = self.conv(x)
        x = self.gn(x)

        return x
