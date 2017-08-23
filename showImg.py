#coding=utf-8
import argparse as ap
import cv2
import numpy as np
import os
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn import preprocessing
import numpy as np
from pylab import *
from PIL import Image

im_features, image_paths, idf, numWords, voc = joblib.load("bof.pkl")
# im_features2 = [[str(num) for num in feat] for feat in im_features]
# for i in range(0, len(im_features2)):
#     ft = ",".join(im_features2[i])
#     print "ft: ", ft
#     line = image_paths[i]+": "+ft
#     with open("im_features.txt", 'a') as wp:
#         wp.write(line+"\n")
#         wp.close()
# im_features = [[float(num) for num in feat] for feat in im_features]
# print im_features
figure()
gray()
subplot(5, 4, 1)
axis('off')
with open("00009-feats.txt", 'r') as rp:
    lines = rp.readlines()
    print lines
    for j in range(20):
        line = lines[j]
        f1 = [float(num) for num in line.replace("\t", " ")[1:].split(",")]
        # print "f1: ", f1
        f2 = np.array([f1], 'float32')
        print "f2: ", f2
        print im_features.dtype
        # f3 = np.zeros((1, numWords), "float32")
        score = np.dot(f2, im_features.T)
        rank_ID = np.argsort(-score)
        for i, ID in enumerate(rank_ID[0][0:]):
            print "ID: ", ID
            img = Image.open(image_paths[ID])
            gray()
            subplot(5, 4, i + 5)
            imshow(img)
            axis('off')
            show()
            break
        # if f1 in im_features:
        #     print "ok"

