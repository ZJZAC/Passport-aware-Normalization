import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
import torch.nn.init as init
from models.layers.fc3_ft import fc3_ft
from torch.autograd import Variable

from models.layers.conv_pub_bn import conv_pub_bn
from models.layers.fc1_pub_bn import fc1_pub_bn

# pointnet_util.py封装着一些重要的函数组件，pointnet.py 或其他用来搭建模型。

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    #用来在ball query过程中确定每一个点距离采样点的距离。
    # C为输入点的通道数（如果是xyz时C = 3），返回的是两组点之间两两的欧几里德距离，即N×M的矩阵。

    # """
    # Calculate Euclid distance between each two points.
    #
    # src^T * dst = xn * xm + yn * ym + zn * zm；
    # sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    # sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    # dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
    #      = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    #
    # Input:
    #     src: source points, [B, N, C]
    #     dst: target points, [B, M, C]
    # Output:
    #     dist: per-point square distance, [B, N, M]
    # """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  #2*(xn * xm + yn * ym + zn * zm)
    dist += torch.sum(src ** 2, -1).view(B, N, 1)  # xn*xn + yn*yn + zn*zn
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)     # xm*xm + ym*ym + zm*zm
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]  S是N的一个子集
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):

    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  #先随机初始化一个centroids矩阵，后面用于存储npoint个采样点的索引位置，大小为B×npoint
    distance = torch.ones(B, N).to(device) * 1e10                 # 利用distance矩阵记录某个样本中所有点到某一个点的距离，初始化为B×N矩阵，初值给个比较大的值，后面会迭代更新;
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)   #利用farthest表示当前最远的点，也是随机初始化，范围为0~N，初始化B个，对应到每个样本都随机有一个初始最远点
    batch_indices = torch.arange(B, dtype=torch.long).to(device)  #batch_indices初始化为0~(B-1)的数组；
    # 建立一个mask，如果dist中的元素小于distance矩阵中保存的距离值，则更新distance中的对应值，随着迭代的继续distance矩阵中的值会慢慢变小，
    # 其相当于记录着某个样本中每个点距离所有已出现的采样点的最小距离；
    for i in range(npoint):
        centroids[:, i] = farthest   # 更新第i个最远点
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3) # 取出这个最远点的xyz坐标
        dist = torch.sum((xyz - centroid) ** 2, -1)  # 计算点集中的所有点到这个最远点的欧式距离
        mask = dist < distance
        distance[mask] = dist[mask]   # 更新distances，记录样本中每个点距离所有已出现的采样点的最小距离
        farthest = torch.max(distance, -1)[1]  # 更新distances，记录样本中每个点距离所有已出现的采样点的最小距离
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    # 函数用于寻找球形领域中的点。输入中radius为球形领域的半径，nsample为每个领域中要采样的点，new_xyz为S个球形领域的中心（由最远点采样在前面得出），xyz为所有的点云；
    # 输出为每个样本的每个球形领域的nsample个采样点集的索引[B, S, nsample]
    # 详细的解析都在备注里。
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)  # sqrdists: [B, S, N] 记录中心点与所有点之间的欧几里德距离
    group_idx[sqrdists > radius ** 2] = N   # 找到所有距离大于radius^2的，其group_idx直接置为N；其余的保留原来的值
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # 做升序排列，前面大于radius^2的都是N，会是最大值，所以会直接在剩下的点中取出前nsample个点
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    # 考虑到有可能前nsample个点中也有被赋值为N的点（即球形区域内不足nsample个点），这种点需要舍弃，直接用第一个点来代替即可
    # group_first: [B, S, k]， 实际就是把group_idx中的第一个点的值复制为了[B, S, K]的维度，便利于后面的替换
    mask = group_idx == N  # 找到group_idx中值等于N的点
    group_idx[mask] = group_first[mask]  # 将这些点的值替换为第一个点的值
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    # Sampling + Grouping主要用于将整个点云分散成局部的group
    # 先用farthest_point_sample函数实现最远点采样FPS得到采样点的索引，再通过index_points将这些点的从原始点中挑出来，作为new_xyz
    # 利用query_ball_point和index_points将原始点云通过new_xyz 作为中心分为npoint个球形区域其中每个区域有nsample个采样点
    # 每个区域的点减去区域的中心值
    # 如果每个点上面有新的特征的维度，则用新的特征与旧的特征拼接，否则直接返回旧的特征
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    torch.cuda.empty_cache()  # 只有执行完这句，显存才会在Nvidia-smi中释放
    new_xyz = index_points(xyz, fps_idx)   # 从原点云中挑出最远点采样的采样点为new_xyz
    torch.cuda.empty_cache()
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # idx:[B, npoint, nsample] 代表npoint个球形区域中每个区域的nsample个采样点的索引
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]  # grouped_xyz:[B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  # grouped_xyz减去采样点即中心值
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    # sample_and_group_all直接将所有点作为一个group
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    # 主要是前面的一些函数的叠加应用。首先先通过sample_and_group的操作形成局部的group，然后对局部的group中的每一个点做MLP操作，最后进行局部的最大池化，得到局部的全局特征。
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    # 大部分的形式都与普通的SA层相似，但是这里radius_list输入的是一个list例如[0.1, 0.2, 0.4][0.1, 0.2, 0.4][0.1, 0.2, 0.4]，对于不同的半径做ball
    # query，最终将不同半径下的点点云特征保存在new_points_list中，再最后拼接到一起。

    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)



def re_initializer_layer(model, num_classes, layer=None):
    """remove the last layer and add a new one"""
    indim = model.fc3.in_features
    private_key = model.fc3
    if layer:
        model.fc3 = layer
    else:
        model.fc3 = nn.Linear(indim, num_classes).cuda()
    return model, private_key

def re_initializer_passport_layer(model, num_classes, layer=None):
    """remove the last layer and add a new one"""
    indim = model.p3.in_features
    private_key = model.p3.key_private
    private_skey = model.p3.skey_private
    private_layer = model.p3
    if layer:
        model.p3 = layer
    else:
        model.p3 = fc3_ft(indim, num_classes).cuda()
        model.p3.key_private  = private_key
        model.p3.skey_private  = private_skey

    return model, private_layer


class STN3d_s_bn(nn.Module):
    def __init__(self, channel):
        super(STN3d_s_bn, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1) #3 or 6
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

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


class STNkd_s_bn(nn.Module):
    def __init__(self, k=64):
        super(STNkd_s_bn, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

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

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class STN3d_gn(nn.Module):
    def __init__(self, channel):
        super(STN3d_gn, self).__init__()
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


class STNkd_gn(nn.Module):
    def __init__(self, k=64):
        super(STNkd_gn, self).__init__()
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

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x



class STN3d_p_bn(nn.Module):
    def __init__(self, channel):
        super(STN3d_p_bn, self).__init__()
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


class STNkd_p_bn(nn.Module):
    def __init__(self, k=64):
        super(STNkd_p_bn, self).__init__()
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