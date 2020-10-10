import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np

main_path_m = "/public/zhangjie/3D/ZJ/simple/datasets/modelnet40_npy/"
main_path_s = "/public/zhangjie/3D/ZJ/simple/datasets/shapenet_npy/"

# main_path_m = "/public/zhangjie/3D/ZJ/simple/datasets/modelnet_trigger/"
# main_path_s = "/public/zhangjie/3D/ZJ/simple/datasets/shapenet_trigger/"


def get_data(train=True,Shapenet=True):
    main_path = main_path_s if Shapenet else main_path_m
    train_txt_path = main_path + "train"
    valid_txt_path = main_path + "val"
    data_txt_path = train_txt_path if train else valid_txt_path
    data = os.listdir(data_txt_path)
    dataNum = len(data)
    clouds_li = []
    labels_li = []
    for i in range(dataNum):
        f = data_txt_path + "/" +data[i]
        pts = np.load(f)
        pts = pts[:1024,:]
        # pts = pts[:512,:]
        lbl = data[i].split(".")[0].split("_")[1]
        lbl = np.array(int(lbl)).reshape(1,)

        clouds_li.append(torch.Tensor(pts).unsqueeze(0))
        labels_li.append(torch.Tensor(lbl).unsqueeze(0))

    clouds = torch.cat(clouds_li)
    labels = torch.cat(labels_li)
    return clouds, labels.long().squeeze()



class PointDataSet(Dataset):
    def __init__(self, train=True,Shapenet=True):
        clouds, labels = get_data(train=train,Shapenet=Shapenet)

        self.x_data = clouds
        self.y_data = labels

        self.lenth = clouds.size(0)
        # print(self.lenth)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # print(f'the legenth of {Dataset} is {self.lenth}')
        return self.lenth


def get_dataLoader(train=True, Shapenet=True, batchsize=16):
    point_data_set = PointDataSet(train=train, Shapenet=Shapenet)
    data_loader = DataLoader(dataset=point_data_set, batch_size=batchsize, shuffle=train)
    return data_loader


def main():

    print("getData_main")

if __name__ == '__main__':
    main()

