
import os
import numpy as np
import random





data_txt_path = "/public/zhangjie/3D/ZJ/simple/datasets/shapenet_trigger/trigger_ori"
data_txt_path2 = "/public/zhangjie/3D/ZJ/simple/datasets/shapenet_trigger/trigger2"
data = os.listdir(data_txt_path)
dataNum = len(data)

for i in range(dataNum):
    f = data_txt_path + "/" +data[i]
    ori_name = data[i].split(".")[0]
    pts = np.load(f)
    # pts = pts[:1024,:]
    lbl = ori_name.split("_")[1]
    new_lbl = (int(lbl)+1)%16  #16 or 40
    print("label",lbl)
    print("new_label",new_lbl)

    new_name = data_txt_path2 + "/" + ori_name + "_" + str(new_lbl)
    np.save(new_name, pts)





def main():

    print("getData_main")

if __name__ == '__main__':
    main()
