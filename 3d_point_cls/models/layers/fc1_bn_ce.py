import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from models.loss.sign_loss import SignLoss


class fc1_bn_ce(nn.Module):
    def __init__(self, i, o):
        super().__init__()

        self.alpha = 0.1
        b = torch.sign(torch.rand(o) - 0.5)  # bit information to store random
        self.register_buffer('b', b)
        self.sign_loss_private = SignLoss(self.alpha, self.b)

        self.linear = nn.Linear(i, o, bias=True)
        self.weight = self.linear.weight
        self.bn0 = nn.BatchNorm1d(o, affine=False)
        self.bn1 = nn.BatchNorm1d(o, affine=False)

        # self.fc = nn.Sequential( )


        self.fc = nn.Sequential(
            nn.Linear(o, o // 4, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(o // 4, o, bias=False),
        )

        # self.fc = nn.Sequential(
        #     nn.Linear(o, o // 4, bias=False),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(o // 4, o, bias=False),
        #     nn.Sigmoid()
        # )


        self.register_buffer('key_private', None)
        self.register_buffer('skey_private', None)

        self.init_scale(True)   #相比v1 此处是true
        self.init_bias(True)    #相比v1 此处是true

        self.reset_parameters()
        self.requires_reset_key = False
        self.key_type = 'random'

    def init_bias(self, force_init=False):
        if force_init:
            self.bias = nn.Parameter(torch.Tensor(self.linear.out_features).to(self.weight.device))
            init.zeros_(self.bias)
        else:
            self.bias = None

    def init_scale(self, force_init=False):
        if force_init:
            self.scale = nn.Parameter(torch.Tensor(self.linear.out_features).to(self.weight.device))
            init.ones_(self.scale)
        else:
            self.scale = None

    def reset_parameters(self):
        init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def set_key(self, x, y=None):   #设置每一层的passport

        self.register_buffer('key_private', x)

        self.register_buffer('skey_private', y)

    def get_scale_key(self):
        return self.skey_private

    def get_scale(self, force_passport=False, ind=0):  #ind=0 ori, ind=1 passport
        if self.scale is not None and not force_passport and ind == 0:
            return self.scale.view(1, -1), self.scale.view(1, -1)
        else:
            skey = self.skey_private
            scale_loss = self.sign_loss_private

            scalekey = self.linear(skey)  #b* out_c
            c = scalekey.size(1)

            scale1 = scalekey.mean(dim=0).view(1, c)
            scale =self.fc(scale1).view(1, c)

            if scale_loss is not None:
                scale_loss.reset()
                scale_loss.add(scale1)

            return scale1, scale


    def get_bias_key(self):
        return self.key_private

    def get_bias(self, force_passport=False, ind=0):
        if self.bias is not None and not force_passport and ind == 0:
            return self.bias.view(1, -1)
        else:
            key = self.key_private

            biaskey = self.linear(key)  # key batch always 1
            c = biaskey.size(1)
            bias = biaskey.mean(dim=0).view(1, c)
            bias =self.fc(bias).view(1, c)

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

        #继承nn.module中的_load_from_state_dict
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
                                      device=x.device))  #x,y both random

        x = self.linear(x)

        x = self.bn0(x)

        # if ind == 0:
        #     x = self.bn0(x)
        # else:
        #     x = self.bn1(x)

        scale1, scale2 = self.get_scale(force_passport, ind)
        x = scale2 * x + self.get_bias(force_passport, ind)
        # x = self.get_scale(force_passport, ind) * x + self.get_bias(force_passport, ind)
        return x
