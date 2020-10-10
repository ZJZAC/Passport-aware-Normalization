import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from models.layers.conv_pub_bn import conv_pub_bn
from models.layers.fc1_pub_bn import fc1_pub_bn

from models.layers.conv_private_bn_ce import conv_private_bn_ce
from models.layers.fc1_bn_ce import fc1_bn_ce
from models.layers.fc2_bn_ce import fc2_bn_ce
from models.layers.fc3_ce import fc3_ce

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = conv_pub_bn(channel, 64, 1) #3 or 6
        self.conv2 = conv_pub_bn(64, 128, 1)
        self.conv3 = conv_pub_bn(128, 1024, 1)
        self.fc1 = fc1_pub_bn(1024, 512)
        self.fc2 = fc1_pub_bn(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x,  ind=0):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x,ind))
        x = F.relu(self.conv2(x,ind))
        x = F.relu(self.conv3(x,ind))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x, ind))
        x = F.relu(self.fc2(x, ind))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3) #变成3*3的矩阵
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = conv_pub_bn(k, 64, 1)
        self.conv2 = conv_pub_bn(64, 128, 1)
        self.conv3 = conv_pub_bn(128, 1024, 1)
        self.fc1 = fc1_pub_bn(1024, 512)
        self.fc2 = fc1_pub_bn(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.k = k

    def forward(self, x,  ind=0):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x,ind))
        x = F.relu(self.conv2(x,ind))
        x = F.relu(self.conv3(x,ind))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x, ind))
        x = F.relu(self.fc2(x, ind))
        x = self.fc3(x)

        #STN3d 就是k=3的特殊形式
        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))  #torch.bmm()三维举着相乘，为了相乘所以才有transpose（2，1）
    return loss

class get_model(nn.Module):
    def __init__(self, k=40, channel=3):
        super(get_model, self).__init__()


        self.stn = STN3d(channel)
        self.convp1 = conv_private_bn_ce(channel,64,1)
        self.fstn = STNkd(k=64)
        self.convp2 = conv_private_bn_ce(64,128,1)
        self.convp3 = conv_private_bn_ce(128,1024,1)
        self.p1 = fc1_bn_ce(1024,512)
        self.p2 = fc2_bn_ce(512,256)
        self.p3 = fc3_ce(256, k)
        self.relu = nn.ReLU()

    def forward(self, x, force_passport=False, ind=0):
        trans = self.stn(x,ind)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.convp1(x, force_passport, ind))

        trans_feat = self.fstn(x,ind)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2, 1)
        x = F.relu(self.convp2(x, force_passport, ind))


        x = self.convp3(x, force_passport, ind)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.p1(x, force_passport, ind))
        x = F.relu(self.p2(x, force_passport, ind))

        x = self.p3(x,force_passport, ind)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat

    def freeze_hidden_layers(self):
        self._freeze_layer(self.stn)
        self._freeze_layer(self.convp1)
        self._freeze_layer(self.fstn)
        self._freeze_layer(self.convp2)
        self._freeze_layer(self.convp3)
        self._freeze_layer(self.p1)
        self._freeze_layer(self.p2)

    def freeze_passport_layers(self):
        self._freeze_layer(self.convp1.bn1)
        self._freeze_layer(self.convp1.fc)
        self._freeze_layer(self.convp2.bn1)
        self._freeze_layer(self.convp2.fc)
        self._freeze_layer(self.convp3.bn1)
        self._freeze_layer(self.convp3.fc)
        self._freeze_layer(self.p1.bn1)
        self._freeze_layer(self.p1.fc)
        self._freeze_layer(self.p2.bn1)
        self._freeze_layer(self.p2.fc)

    def _freeze_layer(self, layer, freeze=True):
        if freeze:
            for p in layer.parameters():
                p.requires_grad = False
        else:
            for p in layer.parameters():
                p.requires_grad = True

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

# class get_loss2(torch.nn.Module):
#     def __init__(self):
#         super(get_loss, self).__init__()
#
#     def forward(self, pred, target):
#         loss = F.nll_loss(pred, target)
#         return loss
