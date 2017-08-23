#coding=utf-8
#图像旋转

import numpy as np
import argparse
import cv2

 # 定义旋转rotate函数
def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # 返回旋转后的图像
    return rotated

# 构造参数解析器
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# 加载图像并显示
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# 将原图旋转不同角度
rotated = rotate(image, 45)
cv2.imshow("Rotated by 45 Degrees", rotated)
rotated = rotate(image, -45)
cv2.imshow("Rotated by -45 Degrees", rotated)
rotated = rotate(image, 90)
cv2.imshow("Rotated by 90 Degrees", rotated)
rotated = rotate(image, -90)
cv2.imshow("Rotated by -90 Degrees", rotated)
rotated = rotate(image, 180)
cv2.imshow("Rotated by 180 Degrees", rotated)
cv2.waitKey(0)