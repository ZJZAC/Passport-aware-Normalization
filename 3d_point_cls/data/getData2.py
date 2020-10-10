import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np

# main_path_m = "/data-x/g10/zhangjie/3D/datasets/modelnet_trigger/"
# main_path_s = "/data-x/g10/zhangjie/3D/datasets/shapenet_trigger/"


main_path_m = "/public/zhangjie/3D/ZJ/simple/datasets/modelnet_trigger/"
main_path_s = "/public/zhangjie/3D/ZJ/simple/datasets/shapenet_trigger/"
# trigger_txt_path = main_path + "trigger3"
# trigger_txt_path = main_path + "trigger2"
# trigger_txt_path = main_path + "trigger"
# trigger_txt_path = main_path + "trigger_m"
# print("FFFFFFFF",trigger_txt_path)

def get_data(Shapenet=True, T1= True):
    if T1:
        trigger_txt_path = main_path_s + "trigger_m" if Shapenet else main_path_m + "trigger_s"
    else:
        trigger_txt_path = main_path_s + "trigger_m2" if Shapenet else main_path_m + "trigger_s2"

    print("Trigger path",trigger_txt_path)
    data = os.listdir(trigger_txt_path)
    dataNum = len(data)
    clouds_li = []
    labels_li = []
    for i in range(dataNum):
        f = trigger_txt_path + "/" +data[i]
        pts = np.load(f)
        pts = pts[:1024,:]
        # pts = pts[:512,:]
        lbl = data[i].split(".")[0].split("_")[2]
        lbl = np.array(int(lbl)).reshape(1,)

        clouds_li.append(torch.Tensor(pts).unsqueeze(0))
        labels_li.append(torch.Tensor(lbl).unsqueeze(0))

    clouds = torch.cat(clouds_li)
    labels = torch.cat(labels_li)
    return clouds, labels.long().squeeze()



class PointDataSet(Dataset):
    def __init__(self,Shapenet=True,T1=True):
        clouds, labels = get_data(Shapenet=Shapenet, T1=T1)

        self.x_data = clouds
        self.y_data = labels

        self.lenth = clouds.size(0)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.lenth


def get_dataLoader( Shapenet=True, T1=True, batchsize=16):
    point_data_set = PointDataSet(Shapenet=Shapenet, T1=T1)
    data_loader = DataLoader(dataset=point_data_set, batch_size=batchsize)
    return data_loader


def main():
    trigger_loader = get_dataLoader()
    print("getData_main")

if __name__ == '__main__':
    main()

