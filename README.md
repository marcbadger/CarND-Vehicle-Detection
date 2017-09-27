# Vehicle Detection Project

Overview
---
This repository contains files for the Vehicle Detection Project. A detailed writeup of the project is given below.

The goals / steps of this project were to:
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[YCrCb_channels]: ./examples/YCrCb_channels.png " "
[HLS_channels]: ./examples/HLS_channels.png " "
[RGB_channels]: ./examples/RGB_channels.png " "
[num_orientations_2_4_8_12]: ./examples/num_orientations_2_4_8_12.png " "
[pix_per_cell_4_8_16]: ./examples/pix_per_cell_4_8_16.JPG " "
[hog_subsampling_detections_test4_allScales]: ./examples/hog_subsampling_detections_test4_allScales.JPG " "
[multi_scale_raw_box_detections]: ./examples/multi_scale_raw_box_detections.png " "
[final_detections_and_heatmaps]: ./examples/final_detections_and_heatmaps.jpg " "
[boxesGIF]: ./examples/output1_boxes.gif "Sliding window detections on a short clip."
[trackedJointGIF]: ./examples/output1_tracked_15_joint_w_time.gif "Polynomial fits to window boxes, lanes fit jointly so they have the same shape in the perspective transformed image."
[trackedSepGIF]: ./examples/output1_tracked_15_sep_w_time.gif "Polynomial fits to window boxes, lanes fit independently so they can have different shapes in the perspective transformed image."
[bouncingGIF]: ./examples/bouncing_example.gif "The perspective transform changes when the car bounces, making decreasing the fit quaility when fitting lines jointly."
[finalResult]: ./examples/output1_tracked_long_15_joint_w_time.gif "Final result, lines fit jointly, allowing parameters to vary linearly over time, using the previous 15 frames of window locations for each frame."
[video1]: ./project_video_tracked.mp4 "Video"
[challengeResult]: ./examples/challenge_tracked.gif "My pipeline didn't do as well on the challenge video!"


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. My project includes the following files:
* [camera_calibration.py](https://github.com/marcbadger/CarND-Advanced-Lane-Lines/blob/master/code/camera_calibration.py) - a Python script to generate camera matrix and distortion coefficients from checkerboard images
* [image_gen.py](https://github.com/marcbadger/CarND-Advanced-Lane-Lines/blob/master/code/image_gen.py) - a Python script to run the tracking pipeline on sample images
* [video_gen.py](https://github.com/marcbadger/CarND-Advanced-Lane-Lines/blob/master/code/video_gen.py) - a Python script to run the tracking pipeline on sample videos
* [tracker_vid.py](https://github.com/marcbadger/CarND-Advanced-Lane-Lines/blob/master/code/tracker_vid.py) - a Python class that handles lane finding and fitting
* [project_video_tracked.mp4](https://github.com/marcbadger/CarND-Advanced-Lane-Lines/blob/master/code/project_video_tracked.mp4) - the output of the lane finding pipeline on the sample video.
* [example output images](https://github.com/marcbadger/CarND-Advanced-Lane-Lines/tree/master/output_images) - intermediate output of image_gen.py
* [README.md, this document](https://github.com/marcbadger/CarND-Advanced-Lane-Lines/blob/master/README.md) - a writeup report summarizing the results

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images (including which color space was chosen, which HOG parameters (orientations, pixels_per_cell, cells_per_block) and why.

The code for this step is contained in lines # through # of the file `XXX.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  See below for an example of one of each of the `vehicle` and `non-vehicle` classes.

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) by selecting random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.  All images below use the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` unless otherwise noted.

Of the three `YCrCb` channels, the first seems to be the most helpful distinguishing cars.  The images below show the first, second, and third channels going down from the top. Cars sort of look like a spiral pattern whereas other "noncar" image patches don't.

![alt text][YCrCb_channels]

For `HLS`, the second channel seemed best.

![alt text][HLS_channels]

For `RGB`, all the channels seemed to be equally helpful, but might be less helpful with a different "notcar" example.

![alt text][RGB_channels]

The `orientations` parameter controlls how many bins the gradients get assigned to.  Having enough orientations to distinguish the rounded corners of cars, e.g. might be important.  At the extreme, if orientations is 4, car's could look just like square barriers. But too many bins would require more `pixels_per_cell` to adequately fill the bins to accurately estimate the distribution of orientations for the gradient. The images below show 2, 4, 8, 12 orientation bins and going down from the top.  Eight bins seems most helpful without requiring a corresponding increase in `pixels_per_cell'.

![alt text][num_orientations_2_4_8_12]

The `pixels_per_cell` parameter controlls how many pixels we use to compute each histogram.  Larger values give courser resolution, but the resulting distribution of sampled gradients is closer to the true distribution. The images below show 4, 8, and 16 pixels along each edge of the cell going down from the top.  Cells with 8 pixel sides seems most helpful because it allows gradients to be accurately estimated and smooths out some of the spatial noise without loosing the overall shape of the car, which happens in the third row where cells are 16 pixels on a side.

![alt text][pix_per_cell_4_8_16]

The cell_per_block parameter, which controlls normalization of histogram counts over local areas ("blocks" of cells) didn't seem to make any visual difference, but it might make a difference for the classifier (see below).

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters (defaults: `color_space=YCrCb, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL') and assessed performance using the accuracy of support vector classification (SVC, see below) and feature extraction time (on all 17760 images in the supplied "cars" and "noncars" sets) on the test images.  Note that because some of these parameters interact, searching along single dimensions might not reach the overall best combination.  A grid search would be good here, but is too computationally intensive.

* Color space: `YCrCb` vs. `HLS` vs. `RGB`
* Orientations: 4 vs. 9 vs. 12
* Pixels per cell: 8 vs. 16
* Cells per normalization block: 1 vs. 2
* HOG channels: 0 vs. 'ALL'
* One more with the best option of each test above

| Parameter     | Extrct/Train Tm(s)| Feature Length| Accuracy   	|
|:-------------:|:-----------------:|:-------------:|:-------------:|
| YCrCb     	| 140/30			|	8412		| 0.9885		|
| HLS     		| 113/31			|	8412		| 0.9904		|
| RGB     		| 122/35			|	8412		| 0.9783		|
| orient:4		| 257/25			|	5472		| 0.9823		|
| orient:9		| 140/30			|	8412		| 0.9885		|
| orient:12		| 203/7				|	10176		| 0.9921, 0.9913, 0.9918		|
| pix_p_cell:8	| 140/30			|	8412		| 0.9885		|
| pix_p_cell:16	| 178/11			|	4092		| 0.9885		|
| cell_p_blok:1	| 128/16			|	4848		| 0.9904		|
| cell_p_blok:2	| 140/30			|	8412		| 0.9885		|
| channel:0		| 62/19				|	4884		| 0.9854		|
| channel:'ALL'	| 140/30			|	8412		| 0.9885		|
| HLS,12,8,1,All| 132/22			|	5424		| 0.9935, 0.9921, 0.9916		|

Based on the above data, I chose to continue with the `HLS` color space with 12 bins, 8 pixels per cell, and one cell per block because it gave the best accuracy. Note that these accuracies are probably artificially high because images are taken from video sequences, but are randomly split into train and test sets.  This makes it very likely that any given test image is very similar to a training image.

Interestingly, when I tried both of these classifiers on the test images and in the actual video pipeline, the `HLS` color space classifier definitely performed much worse and gave more false positives!

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using a feature vector composed of HOG features, raw pixels, and a color histogram.  The HOG features were extracted from all three chanels of the image converted to `YCrCb` color space and were computed on an 8 pixel grid with 12 orientation bins.  I also included raw pixels of a 32 x 32 resized image using the `bin_spatial()` function provided in the lesson.  Finally, I added a feature that concatenated the intensity histogram of each color channel separately.  This gave 10176 features in total.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the HOG subsampling approach in which HOG features are computed for the entire image and then subsampled.  With the reasoning that cars near farther away would be smaller and closer to the horizon, I initially searched over three regions and spatial scales.  I searched the lower half of the image at a scale of 1.5, a slightly smaller region starting halfway down the image at a scale of 1, and the third quarter of the image from the top at a scale of 0.5.  Detections at 1.5, 1, and 0.5 scales are shown in the blue, green, and red, respectively in the image below. 

![alt text][hog_subsampling_detections_test4_allScales]

![alt text][multi_scale_raw_box_detections]

The boxes at the smallest scale seemed to have a lot of false positives and took a long time to evaluate (because there were so many of them), so I decided to only keep the two larger search scales.

Finally, I created a heatmap image, where each box classified as a car added a square of that size to the pixels of the heatmap image.  The heatmap thus accumulates the evidence from all the boxes together, and the result can be thresholded so that single detections are thrown out.  Then by using the `scipy.ndimage.measurements.label()` function, I extracted a bounding box for each separate blob in the heatmap.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I settled on searching at two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector.  The only "false-positive" was a detection in a shadow near the barrier in one image (but in fact, if you watch the video of that section, there's actually a car on the other side of the barrier there).  Here are some example images of heatmaps on the right and the corresponding 	`label()` blob detections:

![alt text][final_detections_and_heatmaps]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Overlapping bounding boxes were handled by the heatmap accumulation method described above.  This method gave a set of non-overlapping bounding boxes for each frame.  Nearby detections in subsequent frames are stored in a list of tracks, where each track has the form: `(status{"new","alive","almostdead","dead"}, numDetectionsInTrack, framesSinceLastSeen, (bbox_age_0, bbox_age_1, ...)`

New bounding boxes from each frame are assigned to existing tracks according to the [Hungarian asisgnment algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm), which assigns tracks to new detections by minimizing a total cost computed from a cost matrix.  As implemented, the cost matrix is the distance between the center of the last assigned box of each track (rows of the cost matrix) and each new box (columns of the cost matrix).

The basic algorighm run for each new frame is as follows:
1. Delete tracks labeled "dead" from the track list
2. Compute the cost matrix for each remaining track and each new detection
3. If a track was assigned a new box and the new box is within a distance threshold:
	a) add one to the number of detections for that track
	b) if the number of detections is sufficient, mark it as "alive" so it will be drawn
	c) reset the frames since last seen to 0
	d) add the assigned box to the existing list of boxes
4. If a track was not assigned a box in the previous step:
	a) add one to frames since last seen
	b) if we haven't seen it in a loger time than we keep tracks for, mark it as "dead"
	c) otherwise, mark it as "almostdead", meaning we can pick it back up if it missed a frame randomly, but we still won't plot it.
5. For each new detections that were not assigned to tracks:
	a) assign it to a new track, of type "new", meaning it won't be drawn until it gets assigned a few more detections.
6. Return the most recent bounding box for each track

Here's an example of the track history for each car, you can see track assignment is consistent.

And here are the tracks for another example segment where one car occludes another:


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One obvious step to make my pipeline more robust is to perform better treatment of the training and test data when training the classifier. Having examples in the test set theat are so near those from the training set might prevent the classifier from generalizing. For the next iteration, I would definitely go through the training images and do some form of screening or hard negative mining.

One weakness of my current implementation is that it most definitely does not run in real-time.  For a real-time implementation, a different method such as semantic segmentaiton with FCNs or [YOLO](https://pjreddie.com/darknet/yolo/) might work better.

One exciting improvement would be to use spline fits of the previous bounding box trajectories for each track to better estimate the "cost" of assigning a new detection at the end of that trajectory. If the track was already on its way to the new_box location (like a car slowly pulling forward), it shouldn't matter far away it WAS, it should matter how much it had to "accelerate" between its last known {location, speed} to get to the proposed new box location.

