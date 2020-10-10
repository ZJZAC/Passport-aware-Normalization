import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1) #3 or 6
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.GroupNorm(64 // 16, 64, affine=True)
        self.bn2 = nn.GroupNorm(128 // 16, 128, affine=True)
        self.bn3 = nn.GroupNorm(1024 // 16, 1024, affine=True)
        self.bn4 = nn.GroupNorm(512 // 16, 512, affine=True)
        self.bn5 = nn.GroupNorm(256 // 16, 256, affine=True)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
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
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.GroupNorm(64 // 16, 64, affine=True)
        self.bn2 = nn.GroupNorm(128 // 16, 128, affine=True)
        self.bn3 = nn.GroupNorm(1024 // 16, 1024, affine=True)
        self.bn4 = nn.GroupNorm(512 // 16, 512, affine=True)
        self.bn5 = nn.GroupNorm(256 // 16, 256, affine=True)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
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
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.t_bn1 = nn.GroupNorm(64 // 16, 64, affine=True)

        self.fstn = STNkd(k=64)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.t_bn2 = nn.GroupNorm(128 // 16, 128, affine=True)

        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.t_bn3 = nn.GroupNorm(1024 // 16, 1024, affine=True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.GroupNorm(512 // 16, 512, affine=True)

        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=0.4)
        self.bn2 = nn.GroupNorm(256 // 16, 256, affine=True)

        self.fc3 = nn.Linear(256, k)
        self.relu = nn.ReLU()

    def forward(self, x):
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.t_bn1(self.conv1(x)))

        trans_feat = self.fstn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2, 1)

        x = F.relu(self.t_bn2(self.conv2(x)))
        x = self.t_bn3(self.conv3(x))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat



class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)  #negative log likelihood loss，和softmax配合使用
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)  #转换矩阵的损失

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


