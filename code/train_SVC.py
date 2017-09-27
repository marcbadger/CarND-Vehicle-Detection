# Marc Badger
# 9.27.17
# Vehicle Detection Project

import os
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lesson_functions import *
import pathlib
import pickle
from sklearn.externals import joblib

basedir = 'E:/Self_Driving_Cars/CarND-Vehicle-Detection/features'
feature_options = os.listdir(basedir)

directoryName = feature_options[0]

# PICK ONE OF THESE TO USE AS FEATURES FOR THE CLASSIFIER:
#directoryName = 'features_YCrCb_9_8_2_ALL_32_16_True_True_True'
#directoryName = 'features_HLS_9_8_2_ALL_32_16_True_True_True'
#directoryName = 'features_RGB_9_8_2_ALL_32_16_True_True_True'
#directoryName = 'features_YCrCb_4_8_2_ALL_32_16_True_True_True'
directoryName = 'features_YCrCb_12_8_2_ALL_32_16_True_True_True'
#directoryName = 'features_YCrCb_9_16_2_ALL_32_16_True_True_True'
#directoryName = 'features_YCrCb_9_8_1_ALL_32_16_True_True_True'
#directoryName = 'features_YCrCb_9_8_2_0_32_16_True_True_True'
#directoryName = 'features_HLS_12_8_1_ALL_32_16_True_True_True'

# load feature parameters from a pickle
dist_pickle = pickle.load( open(basedir + '/' + directoryName + "/feature_params.p", "rb" ) )
X_scaler = dist_pickle["X_scaler"]
color_space = dist_pickle["color_space"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
hog_channel = dist_pickle["hog_channel"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
spatial_feat = dist_pickle["spatial_feat"]
hist_feat = dist_pickle["hist_feat"]
hog_feat = dist_pickle["hog_feat"]

# load features and labels from the joblib files in the directory
scaled_X = joblib.load(basedir + '/' + directoryName + '/scaled_X.pkl')
y = joblib.load(basedir + '/' + directoryName + '/y.pkl')

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
	scaled_X, y, test_size=0.2, random_state=rand_state) # test_size could also be 0.1


# Train a linear SVC
print('Using:',orient,'orientations',pix_per_cell,
	'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Save the trained model
dist_pickle["svc"] = svc
pickle.dump( dist_pickle, open(basedir + '/' + directoryName + "/svc_pickle.p", "wb"))