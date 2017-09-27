# Marc Badger
# 9.27.17
# Vehicle Detection Project

import os
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import time
from lesson_functions import *
from scipy.ndimage.measurements import label

basedir = 'E:/Self_Driving_Cars/CarND-Vehicle-Detection/features'
feature_options = os.listdir(basedir)

#directoryName = feature_options[0]
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
            
            # boxes are ((box left, box top), (box right, box bottom) boxtop < boxbottom, boxleft < boxright
            # which is ("top left corner", "bottom right corner") measured from top of image
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                img_boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw, ytop_draw+win_draw+ystart)))
                heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left:xbox_left+win_draw] += 1

    print(time.time()-t, 'seconds to run, total windows = ', count)

    return draw_img, heatmap, img_boxes


################################
# MULTIPLE DETECTIONS AND FALSE POSITIVES
################################

# Search for cars at three spatial scales over three regions of the image
# ystarts = [400, 400, 400]
# ystops = [656, 528, 464]
# scales = [1.5, 1, 0.5]
# cells_per_steps = [2, 2, 8]

ystarts = [400, 400]
ystops = [656, 528]
scales = [1.5, 1]
cells_per_steps = [2, 2]

# Load the test images
basedir = 'E:/Self_Driving_Cars/CarND-Vehicle-Detection/test_images'
# load in some test images:
images = glob.glob(basedir + '/test*.jpg')


for idx, fname in enumerate(images):

    filename = fname.split('\\')[-1]

    # load the image
    img = mpimg.imread(fname)
    draw_raw = np.copy(img)
    draw_img = np.copy(img)

    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    box_list = []
    
    counter = 0
    plotColors = ((0,0,255), (0,255,0), (255,0,0))
    # Accumulate the boxes from all spatial scales
    for ystart, ystop, scale, cells_per_step in zip(ystarts, ystops, scales, cells_per_steps):
        
        out_img, heatmap, img_boxes = find_cars(img, ystart, ystop, scale, cells_per_step, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        
        box_list.extend(img_boxes)

        for box in img_boxes:
            aone = box[0]
            atwo = box[1]
            cv2.rectangle(draw_raw,aone,atwo,plotColors[counter],6)

        counter += 1

    # Add to the heatmap for each detected box
    heat = add_heat(heat, box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    fig = plt.figure()
    plt.subplot(221)
    plt.imshow(draw_raw)
    plt.title('Raw detections')
    plt.subplot(222)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    plt.subplot(223)
    plt.imshow(labels[0], cmap='gray')
    plt.title('Labels')
    plt.subplot(224)
    plt.imshow(draw_img)
    plt.title('Car Positions')
    fig.tight_layout()
    write_name = 'C:/Users/marcbadger.MBCOMP/Google Drive/Self_Driving_Cars/CarND-Vehicle-Detection-master/output_images' +'/raw_heatmap_detections_comparison_' + filename[:-4] +'.jpg'
    plt.savefig(write_name, bbox_inches='tight')
    plt.close()

    plt.imshow(draw_img)
    write_name = 'C:/Users/marcbadger.MBCOMP/Google Drive/Self_Driving_Cars/CarND-Vehicle-Detection-master/output_images' +'/hog_subsampling_refined_detections_' + filename[:-4] +'.jpg'
    plt.savefig(write_name, bbox_inches='tight')
    plt.close()

    plt.imshow(draw_raw)
    write_name = 'C:/Users/marcbadger.MBCOMP/Google Drive/Self_Driving_Cars/CarND-Vehicle-Detection-master/output_images' +'/hog_subsampling_raw_detections_' + filename[:-4] +'.jpg'
    plt.savefig(write_name, bbox_inches='tight')
    plt.close()

    plt.imshow(heatmap)
    write_name = 'C:/Users/marcbadger.MBCOMP/Google Drive/Self_Driving_Cars/CarND-Vehicle-Detection-master/output_images' +'/hog_subsampling_refined_heatmap_' + filename[:-4] +'.jpg'
    plt.savefig(write_name, bbox_inches='tight')
    plt.close()


# Initially, assign tracks just based on location alone:
# 1. implement cost matrix
# 2. implement hungarian algorithm

# FROM: https://stackoverflow.com/questions/32046582/spline-with-constraints-at-border

import numpy as np
from scipy.interpolate import UnivariateSpline, splev, splrep
from scipy.optimize import minimize

def guess(x, y, k, s, w=None):
    """Do an ordinary spline fit to provide knots"""
    return splrep(x, y, w, k=k, s=s)

def err(c, x, y, t, k, w=None):
    """The error function to minimize"""
    diff = y - splev(x, (t, c, k))
    if w is None:
        diff = np.einsum('...i,...i', diff, diff)
    else:
        diff = np.dot(diff*diff, w)
    return np.abs(diff)

def spline_neumann(x, y, k=3, s=0, w=None):
    t, c0, k = guess(x, y, k, s, w=w)
    x0 = x[0] # point at which zero slope is required
    con = {'type': 'eq',
           'fun': lambda c: splev(x0, (t, c, k), der=1),
           #'jac': lambda c: splev(x0, (t, c, k), der=2) # doesn't help, dunno why
           }
    opt = minimize(err, c0, (x, y, t, k, w), constraints=con)
    copt = opt.x
    return UnivariateSpline._from_tck((t, copt, k))

import matplotlib.pyplot as plt

n = 10
x = np.linspace(0, 2*np.pi, n)
y0 = np.cos(x) # zero initial slope
std = 0.5
noise = np.random.normal(0, std, len(x))
y = y0 + noise
k = 3

sp0 = UnivariateSpline(x, y, k=k, s=n*std)
sp = spline_neumann(x, y, k, s=n*std)

plt.figure()
X = np.linspace(x.min(), x.max(), len(x)*10)
plt.plot(X, sp0(X), '-r', lw=1, label='guess')
plt.plot(X, sp(X), '-r', lw=2, label='spline')
plt.plot(X, sp.derivative()(X), '-g', label='slope')
plt.plot(x, y, 'ok', label='data')
plt.legend(loc='best')
plt.show()