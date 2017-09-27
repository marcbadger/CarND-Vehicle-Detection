# Marc Badger
# 9.27.17
# Vehicle Detection Project

import os
import glob
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import time
from lesson_functions import *
from scipy.ndimage.measurements import label
from vehicles import Vehicles
import random

basedir = 'E:/Self_Driving_Cars/CarND-Vehicle-Detection/features'
feature_options = os.listdir(basedir)

# PICK ONE OF THESE TO USE AS A CLASSIFIER:
#directoryName = feature_options[0]
#directoryName = 'features_HLS_12_8_1_ALL_32_16_True_True_True'
# directoryName = 'features_YCrCb_12_8_2_ALL_32_16_True_True_True'
#directoryName = 'features_YCrCb_9_8_2_ALL_32_16_True_True_True'
directoryName = 'features_YCrCb_8_8_2_ALL_32_16_True_True_True'

# load feature parameters from a pickle
dist_pickle = pickle.load( open(basedir + '/' + directoryName + "/svc_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
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

# Define a single function that can extract features using hog sub-sampling and make predictions
# Allows us to only extract hog features once and then we sample these features to get inputs for the classifier
# Each window defined by a scaling factor.  Scale = 1 => window that is 8 x 8 cells
# Overlap is in terms of cell distance, so cells_per_step = 2 would be 75% overlap for an 8 x 8 cell size.
# Run this function more than once with different ystart, ystop, and scales to gather multiple-scaled search windows.
def find_cars(img, ystart, ystop, scale, cells_per_step, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    img_boxes = []
    t=time.time()
    count = 0

    draw_img = np.copy(img)
    heatmap = np.zeros_like(img[:,:,0])
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    # Don't forget to convert to the color format you USED DURING TRAINING!
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1 # WHY - cell_per_block?
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 # WHY - cell_per_block?
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1 # WHY - cell_per_block?
    #cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            count += 1
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                img_boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw, ytop_draw+win_draw+ystart)))
                heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left:xbox_left+win_draw] += 1

    #print(time.time()-t, 'seconds to run, total windows = ', count)

    return draw_img, heatmap, img_boxes


################################
# MULTIPLE DETECTIONS AND FALSE POSITIVES
################################

# Search for cars at two-three spatial scales over three regions of the image
# ystarts = [400, 400, 400]
# ystops = [656, 528, 464]
# scales = [1.5, 1, 0.5]
# cells_per_steps = [2, 2, 4]

ystarts = [400, 400]
ystops = [656, 528]
scales = [1.5, 1]
cells_per_steps = [2, 2]
heatmapThreshold = 1

# Load the test images
basedir = 'E:/Self_Driving_Cars/CarND-Vehicle-Detection/test_images'
# load in some test images:
images = glob.glob(basedir + '/test*.jpg')


def process_image(img):

    draw_img = np.copy(img)
    draw_overlay = cv2.cvtColor(np.copy(img), cv2.COLOR_RGB2HSV)

    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    box_list = []
    
    # Accumulate the boxes from all spatial scales
    for ystart, ystop, scale, cells_per_step in zip(ystarts, ystops, scales, cells_per_steps):
        
        out_img, heatmap, img_boxes = find_cars(img, ystart, ystop, scale, cells_per_step, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        
        box_list.extend(img_boxes)
    
    # Add to the heatmap for each detected box
    heat = add_heat(heat, box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, heatmapThreshold)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    labeled_boxes = []

    for detection_number in range(1, labels[1]+1):
        # Find pixels with each detection_number label value
        nonzero = (labels[0] == detection_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        labeled_boxes.append(((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))))
    

    boxesToReturn = 'avgd' # Can be 'avgd', 't_hist', or 'raw' to plot the average box, a time history, or raw detected boxes
    
    if boxesToReturn == 'avgd':
        # Calculate the new box locations and plot them
        detected_car_boxes = vehicles.assign_detections(labeled_boxes, num_history=15, raw_or_avgd='avgd')
        for bbox in detected_car_boxes:
            # Draw the box on the image
            cv2.rectangle(draw_img, tuple(bbox[0]), tuple(bbox[1]), (0,0,255), 6)
    elif boxesToReturn == 't_hist':
        detected_car_boxes = vehicles.assign_detections(labeled_boxes, num_history=15, raw_or_avgd='raw')
        hues = np.linspace(1,255, 30)
        hsvColors = np.array([(h, 255, 255) for h in hues])
        for idx, track in enumerate(detected_car_boxes):
            for bbox in track:
                # Draw the box on the image
                cv2.rectangle(draw_overlay, tuple(bbox[0]), tuple(bbox[1]), hsvColors[idx], 6)

        draw_overlay = cv2.cvtColor(draw_overlay, cv2.COLOR_HSV2RGB)
        cv2.addWeighted(draw_overlay, 0.6, draw_img, 0.8, 0, draw_img)
    else:
        detected_car_boxes = labeled_boxes
        for bbox in detected_car_boxes:
            # Draw the box on the image
            cv2.rectangle(draw_img, tuple(bbox[0]), tuple(bbox[1]), (0,0,255), 6)

    frameNumber = vehicles.frames_analyzed
    carsInView = len(detected_car_boxes)

    cv2.putText(draw_img, 'Frame number ' + str(frameNumber), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(draw_img, 'Num cars in view ' + str(carsInView), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

    return draw_img

    # fig = plt.figure()
    # plt.subplot(121)
    # plt.imshow(draw_img)
    # plt.title('Car Positions')
    # plt.subplot(122)
    # plt.imshow(heatmap, cmap='hot')
    # plt.title('Heat Map')
    # fig.tight_layout()
    # write_name = 'C:/Users/marcbadger.MBCOMP/Google Drive/Self_Driving_Cars/CarND-Vehicle-Detection-master/output_images' +'/detections_and_heatmap_comparison' + filename[:-4] +'.jpg'
    # plt.savefig(write_name, bbox_inches='tight')
    # plt.close()

    # plt.imshow(draw_img)
    # write_name = 'C:/Users/marcbadger.MBCOMP/Google Drive/Self_Driving_Cars/CarND-Vehicle-Detection-master/output_images' +'/hog_subsampling_refined_detections_' + filename[:-4] +'.jpg'
    # plt.savefig(write_name, bbox_inches='tight')
    # plt.close()

    # plt.imshow(heatmap)
    # write_name = 'C:/Users/marcbadger.MBCOMP/Google Drive/Self_Driving_Cars/CarND-Vehicle-Detection-master/output_images' +'/hog_subsampling_refined_heatmap_' + filename[:-4] +'.jpg'
    # plt.savefig(write_name, bbox_inches='tight')
    # plt.close()


Output_video = 'output1_tracked_new_' + directoryName[9:23] + '_40_5_12.mp4'
Input_video = 'project_video.mp4'

frameNumber = 1

# We only want to initialize the tracker once!
# use the tracker_vid class (see tracker_vid.py) to assign new detections to tracked cars:

# Input to tracker: detected boxes taken from the heatmap from each frame
# Parameters for tracking and assignment:
# Max distance new box can be from any previous track before being assigned as a new car.
# Number of frames a track needs to exist for in order to be drawn on the image (> 2 ish?, which will cut down on single frame false positives).
# Number of frames a track can exist before being killed (probably kill immediately unless false negatives are a problem).

# Initial image:
# Assign a track to each detection, but don't draw anything yet.

# Tracks can be "new", "alive", "almostdead" or "dead"
# "new" if ageSinceBirth < drawThresh # THESE CAN BE ASSIGNED NEW DETECTIONS BUT ARE NOT PLOTTED
# "alive" if ((ageSinceBirth >= drawThresh) & ( ageSinceLastSeen <= 0)) # THESE CAN BE ASSIGNED NEW DETECTIONS AND ARE PLOTTED
# "almostdead" if ((ageSinceLastSeen > 0) & (ageSinceLastSeen <= ageKillThresh)) # THESE CAN BE ASSIGNED NEW DETECTIONS BUT ARE NOT PLOTTED
# "dead" if ageSinceLastSeen > ageKillThresh # THESE CANNOT BE ASSIGNED NEW DETECTIONS AND ARE NOT PLOTTED

# Track objects are (status{"new","alive","almostdead","dead"}, ageSinceBirth, ageSinceLastSeen, {bbox age 1, bbox age 2, ...})

# INITIALIZE THE VEHICLES TO START THE TRACKER
vehicles = Vehicles(MydistThresh = 40, MydrawThresh = 5, MyageKillThresh = 12)

import sys, traceback

try:
    clip1 = VideoFileClip(Input_video)
    video_clip = clip1.fl_image(process_image) # This function expects color images.
    video_clip.write_videofile(Output_video, audio=False)
    del clip1.reader
    del clip1
except:
    tb = traceback.format_exc()
else:
    tb = "Complete!"
finally:
    print(tb)

exit()