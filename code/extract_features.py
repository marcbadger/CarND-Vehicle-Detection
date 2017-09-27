# Marc Badger
# 9.27.17
# Vehicle Detection Project

import os
import glob

###### GET FILENAMES ######
basedir = 'E:/Self_Driving_Cars/CarND-Vehicle-Detection/vehicles/'

image_types = os.listdir(basedir)
cars = []
for imtype in image_types:
	cars.extend(glob.glob(basedir+imtype+'/*'))

print('Number of Vechicle Images found:', len(cars))
with open("cars.txt", 'w') as f:
	for fn in cars:
		f.write(fn+'\n')

basedir = 'E:/Self_Driving_Cars/CarND-Vehicle-Detection/non-vehicles/'

image_types = os.listdir(basedir)
notcars = []
for imtype in image_types:
	notcars.extend(glob.glob(basedir+imtype+'/*'))

print('Number of Non-Vechicle Images found:', len(notcars))
with open("cars.txt", 'w') as f:
	for fn in notcars:
		f.write(fn+'\n')


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

# Define a function for plotting multiple images [TAKEN FROM PROJECT WALKTHROUGH]
def visualize(fig, rows, cols, imgs, titles):
	for i, img in enumerate(imgs):
		plt.subplot(rows, cols, i+1)
		plt.title(i+1)
		img_dims = len(img.shape)
		if img_dims < 3:
			plt.imshow(img, cmap='hot')
			plt.title(titles[i])
		else:
			plt.imshow(img)
			plt.title(titles[i])
	plt.show()

# Parameters for extracting features
color_space = 'YCrCb' # Also tried 'HLS' and 'RGB'
orient = 12 # HOG orientations, also tried 8, 9, 12
pix_per_cell = 8 # HOG pixels per cell, also tried 8 and 16
cell_per_block = 1 # HOG cells per block, also tried 2
hog_channel = 'ALL' #'ALL' # Can be 0, 1, 2, or "ALL". 'ALL' works best when actually extracting features, but can't be used with vis=True
spatial_size = (32,32) #(16,16) # Spatial binning dimensions
hist_bins = 16 #32 # Number of histogram bins
spatial_feat = True # Return spatial features
hist_feat = True # Return histogram features
hog_feat = True # Return HOG features

# Uncomment these lines to plot some example features!
# car_ind = np.random.randint(0,len(cars))
# notcar_ind = np.random.randint(0,len(notcars))

# car_image = mpimg.imread(cars[car_ind])
# notcar_image = mpimg.imread(notcars[notcar_ind])
# car_features, car_hog_image = single_img_features(car_image, color_space = color_space, spatial_size = spatial_size,
# 	hist_bins = hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
# 	spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat, vis=True)

# notcar_features, notcar_hog_image = single_img_features(notcar_image, color_space = color_space, spatial_size = spatial_size,
# 	hist_bins = hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
# 	spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat, vis=True)

# images = [car_image, car_hog_image, notcar_image, notcar_hog_image]
# titles = ['car image', 'car HOG image', 'notcar image', 'notcar HOG image']
# fig = plt.figure(figsize=(12,3))
# visualize(fig, 1, 4, images, titles)


##########################################
# Extracting and saving features to files:
# We'll save the features in a directory named using the hyperparameter configuration
directoryName = 'E:/Self_Driving_Cars/CarND-Vehicle-Detection/features/' + \
				'features_' + '_'.join([color_space, str(orient), str(pix_per_cell), \
					str(cell_per_block), str(hog_channel), str(spatial_size[0]), \
					str(hist_bins),str(spatial_feat), str(hist_feat), str(hog_feat)])

# Create the directory
pathlib.Path(directoryName).mkdir(parents=True, exist_ok = True)

t=time.time()
n_samples = 1000
random_idxs = np.random.randint(0,len(cars),n_samples)
test_cars = cars #np.array(cars)[random_idxs] # TO TRAIN ON FULL DATASET: cars
test_notcars = notcars #np.array(notcars)[random_idxs] # TO TRAIN ON FULL DATASET: notcars

car_features = extract_features(test_cars, color_space=color_space, 
						spatial_size=spatial_size, hist_bins=hist_bins, 
						orient=orient, pix_per_cell=pix_per_cell, 
						cell_per_block=cell_per_block, 
						hog_channel=hog_channel, spatial_feat=spatial_feat, 
						hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(test_notcars, color_space=color_space, 
						spatial_size=spatial_size, hist_bins=hist_bins, 
						orient=orient, pix_per_cell=pix_per_cell, 
						cell_per_block=cell_per_block, 
						hog_channel=hog_channel, spatial_feat=spatial_feat, 
						hist_feat=hist_feat, hog_feat=hog_feat)

print(time.time()-t, 'Seconds to compute features...')

# Concatenate all the features together
X = np.vstack((car_features, notcar_features)).astype(np.float64)						
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Save the computed features for loading later
joblib.dump(scaled_X, directoryName + '/scaled_X.pkl' )
joblib.dump(y, directoryName + '/y.pkl' )

dist_pickle = {}
dist_pickle["X_scaler"] = X_scaler
dist_pickle["color_space"] = color_space
dist_pickle["orient"] = orient
dist_pickle["pix_per_cell"] = pix_per_cell
dist_pickle["cell_per_block"] = cell_per_block
dist_pickle["hog_channel"] = hog_channel
dist_pickle["spatial_size"] = spatial_size
dist_pickle["hist_bins"] = hist_bins
dist_pickle["spatial_feat"] = spatial_feat
dist_pickle["hist_feat"] = hist_feat
dist_pickle["hog_feat"] = hog_feat
pickle.dump( dist_pickle, open(directoryName + "/feature_params.p", "wb"))

# USE THIS TO CREATE A FEATURE VISUALIZATION
car_ind = np.random.randint(0,len(cars))
notcar_ind = np.random.randint(0,len(notcars))
if len(car_features) > 0:
	car_ind = np.random.randint(0, len(cars))
	# Plot an example of raw and scaled features
	fig = plt.figure(figsize=(12,4))
	plt.subplot(131)
	plt.imshow(mpimg.imread(cars[car_ind]))
	plt.title('Original Image')
	plt.subplot(132)
	plt.plot(X[car_ind])
	plt.title('Raw Features')
	plt.subplot(133)
	plt.plot(scaled_X[car_ind])
	plt.title('Normalized Features')
	fig.tight_layout()
	write_name = 'C:/Users/marcbadger.MBCOMP/Google Drive/Self_Driving_Cars/CarND-Vehicle-Detection-master/output_images' +'/scaled_features_comparison_'+str(car_ind) + '.jpg'
	plt.savefig(write_name, bbox_inches='tight')
	plt.close()
else: 
	print('Your function only returns empty feature vectors...')