
import os
import numpy as np
import random

both_m=[0,7,8,17,19,20]

data_txt_path = "/public/zhangjie/3D/ZJ/simple/datasets/modelnet_trigger/test"
data_txt_path2 = "/public/zhangjie/3D/ZJ/simple/datasets/shapenet_trigger/trigger_m"

if not os.path.exists(data_txt_path2):
    os.makedirs(data_txt_path2)
data = os.listdir(data_txt_path)
dataNum = len(data)

for i in range(150):
    f = data_txt_path + "/" +data[i]
    ori_name = data[i].split(".")[0]
    pts = np.load(f)
    # pts = pts[:1024,:]
    lbl = ori_name.split("_")[1]
    lbl = int(lbl)
    if not lbl in both_m:
        new_lbl = lbl
        print("label",lbl)
        print("new_label",new_lbl)
        while new_lbl == lbl :
            new_lbl = random.randint(0,15) #15 or 39
            print("2nd_new_label", new_lbl)
        new_name = data_txt_path2 + "/" + ori_name + "_" + str(new_lbl)
        np.save(new_name, pts)
#

# both_s=[0,3,4,6,8,9,15]
#
# data_txt_path = "/public/zhangjie/3D/ZJ/simple/datasets/shapenet_trigger/test"
# data_txt_path2 = "/public/zhangjie/3D/ZJ/simple/datasets/modelnet_trigger/trigger_s"
#
# if not os.path.exists(data_txt_path2):
#     os.makedirs(data_txt_path2)
# data = os.listdir(data_txt_path)
# dataNum = len(data)
#
# for i in range(dataNum):
#     f = data_txt_path + "/" +data[i]
#     ori_name = data[i].split(".")[0]
#     pts = np.load(f)
#     # pts = pts[:1024,:]
#     lbl = ori_name.split("_")[1]
#     lbl = int(lbl)
#     if not lbl in both_s:
#         new_lbl = lbl
#         print("label",lbl)
#         print("new_label",new_lbl)
#         while new_lbl == lbl :
#             new_lbl = random.randint(0,39) #15 or 39
#             print("2nd_new_label", new_lbl)
#         new_name = data_txt_path2 + "/" + ori_name + "_" + str(new_lbl)
#         np.save(new_name, pts)


def main():

    print("getData_main")

if __name__ == '__main__':
    main()
