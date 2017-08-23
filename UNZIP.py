#coding=utf-8
import os
import zipfile


def Unzip(target_name):
    zipfiles=zipfile.ZipFile(target_name, 'r')
    zipfiles.extractall(os.path.splitext(target_name)[0])
    zipfiles.close()
    print(target_name + " Unzip finished!")

if __name__ == "__main__":
    path = "D:/wudechao/paper/三维模型数据/实验数据/"
    # Unzip(savePath, path)

    files = os.listdir(path=path)
    for file in files:
        if "NTU3D.v1" in file:
            filepath = path+file
            print(filepath)
            Unzip(filepath)
