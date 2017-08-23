#coding=utf-8

import argparse as ap
import cv2

# import imutils
import numpy as np
import os
from sklearn.externals import joblib
from scipy.cluster.vq import *

from sklearn import preprocessing
import numpy as np

from pylab import *
from PIL import Image
# from rootsift import RootSIFT

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-i", "--image", help="Path to query image", required="True")
args = vars(parser.parse_args())

# Get query image path
image_path = args["image"]

# Load the classifier, class names, scaler, number of clusters and vocabulary
im_features, image_paths, idf, numWords, voc = joblib.load("bof.pkl")
print "im_features: ", im_features
print "image_paths: ", image_paths
# Create feature extraction and keypoint detector objects
fea_det = cv2.FeatureDetector_create("SIFT")
des_ext = cv2.DescriptorExtractor_create("SIFT")

# List where all the descriptors are stored
des_list = []

im = cv2.imread(image_path)
kpts = fea_det.detect(im)
kpts, des = des_ext.compute(im, kpts)

# rootsift
#rs = RootSIFT()
#des = rs.compute(kpts, des)

des_list.append((image_path, des))

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]

#
test_features = np.zeros((1, numWords), "float32")
print "test_features: ", test_features
words, distance = vq(descriptors, voc)
for w in words:
    test_features[0][w] += 1

# Perform Tf-Idf vectorization and L2 normalization
test_features = test_features*idf
test_features = preprocessing.normalize(test_features, norm='l2')

print "test_features hou: ", test_features
print "test_features.dtype: ", test_features.dtype
score = np.dot(test_features, im_features.T)
rank_ID = np.argsort(-score)
print "type(rank_ID): ", type(rank_ID)
print "type(rank_ID[0]): ", type(rank_ID[0])
print im_features.dtype
# Visualize the results
figure()
gray()
subplot(5, 4, 1)
imshow(im[:, :, ::-1])
axis('off')
for i, ID in enumerate(rank_ID[0][0:16]):
    print "ID: ", ID
    img = Image.open(image_paths[ID])
    gray()
    subplot(5, 4, i+5)
    imshow(img)
    axis('off')

show()