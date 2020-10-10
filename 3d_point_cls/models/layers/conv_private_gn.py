import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from models.loss.sign_loss import SignLoss


class conv_private_gn(nn.Module):
    def __init__(self, i, o, ks=1):
        super().__init__()

        self.alpha = 0.1
        b = torch.sign(torch.rand(o) - 0.5)  # bit information to store random
        self.register_buffer('b', b)
        self.sign_loss_private = SignLoss(self.alpha, self.b)

        self.conv = nn.Conv1d(i, o, ks,  bias=False)
        self.gn = nn.GroupNorm(o // 16, o, affine=False)
        self.weight = self.conv.weight

        self.register_buffer('key_private', None)
        self.register_buffer('skey_private', None)

        self.init_scale(True)
        self.init_bias(True)

        self.reset_parameters()
        self.requires_reset_key = False
        self.key_type = 'random'

    def init_bias(self, force_init=False):
        if force_init:
            self.bias = nn.Parameter(torch.Tensor(self.conv.out_channels).to(self.weight.device))
            init.zeros_(self.bias)
        else:
            self.bias = None

    def init_scale(self, force_init=False):
        if force_init:
            self.scale = nn.Parameter(torch.Tensor(self.conv.out_channels).to(self.weight.device))
            init.ones_(self.scale)
        else:
            self.scale = None

    def reset_parameters(self):
        init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')


    def set_key(self, x, y=None):

        self.register_buffer('key_private', x)
        self.register_buffer('skey_private', y)

    def get_scale_key(self):
        return self.skey_private

    def get_scale(self, force_passport=False, ind=0):
        if self.scale is not None and not force_passport and ind == 0:
            return self.scale.view(1, -1, 1)
        else:
            skey = self.skey_private
            scale_loss = self.sign_loss_private

            scalekey = self.conv(skey)
            b = scalekey.size(0)
            c = scalekey.size(1)
            scale = scalekey.view(b, c, -1).mean(dim=2).view(b, c, 1)
            scale = scale.mean(dim=0).view(1, c, 1)

            if scale_loss is not None:
                scale_loss.reset()
                scale_loss.add(scale)

            return  scale

    def get_bias_key(self):
        return self.key_private

    def get_bias(self, force_passport=False, ind=0):
        if self.bias is not None and not force_passport and ind == 0:
            return self.bias.view(1, -1, 1)
        else:
            key = self.key_private
            biaskey = self.conv(key)  # key batch always 1
            b = biaskey.size(0)
            c = biaskey.size(1)
            bias = biaskey.view(b, c, -1).mean(dim=2).view(b, c, 1)
            bias = bias.mean(dim=0).view(1, c, 1)

            return bias

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        keyname = prefix + 'key_private'
        skeyname = prefix + 'skey_private'

        if keyname in state_dict:
            self.register_buffer('key_private', torch.randn(*state_dict[keyname].size()))
        if skeyname in state_dict:
            self.register_buffer('skey_private', torch.randn(*state_dict[skeyname].size()))

        scalename = prefix + 'scale'
        biasname = prefix + 'bias'
        if scalename in state_dict:
            self.scale = nn.Parameter(torch.randn(*state_dict[scalename].size()))

        if biasname in state_dict:
            self.bias = nn.Parameter(torch.randn(*state_dict[biasname].size()))

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)



    def generate_key(self, *shape):
        newshape = list(shape)
        newshape[0] = 1

        min = -1.0
        max = 1.0
        key = np.random.uniform(min, max, newshape)
        return key

    def forward(self, x, force_passport=False, ind=0):
        key = self.key_private  #与v1不同之处
        if (key is None and self.key_type == 'random') or self.requires_reset_key:
            self.set_key(torch.tensor(self.generate_key(*x.size()),
                                      dtype=x.dtype,
                                      device=x.device),
                         torch.tensor(self.generate_key(*x.size()),
                                      dtype=x.dtype,
                                      device=x.device))

        x = self.conv(x)
        x = self.gn(x)

        x = self.get_scale(force_passport, ind) * x + self.get_bias(force_passport, ind)
        return x
