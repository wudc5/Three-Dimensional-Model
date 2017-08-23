#coding=utf-8
#python findFeatures.py -t dataset/train/

import argparse as ap
import cv2
import numpy as np
import os
from sklearn.externals import joblib
from scipy.cluster.vq import *

from sklearn import preprocessing
# from rootsift import RootSIFT
# import math

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
args = vars(parser.parse_args())

# Get the training classes names and store them in a list
train_path = args["trainingSet"]
#train_path = "dataset/train/"

training_names = os.listdir(train_path)

numWords = 900

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
for training_name in training_names:
    image_path = os.path.join(train_path, training_name)
    image_paths += [image_path]

print("image_paths: ", image_paths)
# Create feature extraction and keypoint detector objects
fea_det = cv2.FeatureDetector_create("SIFT")
des_ext = cv2.DescriptorExtractor_create("SIFT")

# List where all the descriptors are stored
des_list = []

for i, image_path in enumerate(image_paths):
    im = cv2.imread(image_path)
    print("Extract SIFT of %s image, %d of %d images" %(training_names[i], i, len(image_paths)))

    kpts = fea_det.detect(im)
    kpts, des = des_ext.compute(im, kpts)
    print("des: ", des)
    # print("len(des[0]): ", len(des[0]))
    # rootsift
    #rs = RootSIFT()
    #des = rs.compute(kpts, des)
    if kpts is None or des is None:
        continue
    des_list.append((image_path, des))

print("des_list: ", des_list)
# Stack all the descriptors vertically in a numpy array
#downsampling = 1
#descriptors = des_list[0][1][::downsampling,:]
#for image_path, descriptor in des_list[1:]:
#    descriptors = np.vstack((descriptors, descriptor[::downsampling,:]))

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
print("descriptors: ", descriptors)
for image_path, descriptor in des_list[1:]:
    if descriptor is None:
        continue
    print "image_path: ", image_path
    print "descriptor: ", descriptor
    print "len(descriptor): ", len(descriptor)
    descriptors = np.vstack((descriptors, descriptor))   # np.vstack用来连接

print("descriptors: ", descriptors)
# Perform k-means clustering
print("Start k-means: %d words, %d key points" % (numWords, descriptors.shape[0]))
voc, variance = kmeans(descriptors, numWords, 1)    # voc为聚类后的中心点， varience为方差

print("voc: ", voc)
print("variance: ", variance)
# Calculate the histogram of features
im_features = np.zeros((len(image_paths), numWords), "float32")
for i in range(len(image_paths)):
    if i >= len(des_list) or len(des_list[i]) < 2:
        continue
    words, distance = vq(des_list[i][1], voc)  # 把一副图片的每一维特征归类到距离最近的中心点
    for w in words:
        print("w: ", w)
        im_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Perform L2 normalization
im_features = im_features*idf
im_features = preprocessing.normalize(im_features, norm='l2')
print "im_features: ", im_features
for feat in im_features:
    print "type(feat): ", type(feat)
    line = ""
    for f in feat:
        line = line + str(f) + ","
    line = line[0:len(line)-1]
    print "line: ", line
    with open("features2.txt", 'a') as wp:
        wp.write(line+"\n")
        wp.close()

joblib.dump((im_features, image_paths, idf, numWords, voc), "bof.pkl", compress=3)