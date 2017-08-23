#coding=utf-8
import os
import shutil
if __name__ == "__main__":
    path = "D:/wudechao/paper/三维模型数据/实验数据/"
    files = os.listdir(path)
    JPGFileSavePath = "D:/wudechao/paper/三维模型数据/实验数据/JPGPIC/"
    if not os.path.exists(JPGFileSavePath):
        os.makedirs(JPGFileSavePath)
    for file in files:
        if os.path.isdir(path+file) and "NTU3D.v1" in file:
            pics = os.listdir(path+file)
            for pic in pics:
                if "jpg" in pic:
                    shutil.copyfile(path+file+"/"+pic, JPGFileSavePath+pic)
            print(path + file)
