# Marc Badger
# 9.25.17
# Vehicle Detection Project

import numpy as np
import cv2
from scipy.optimize import curve_fit, linear_sum_assignment
import matplotlib.pyplot as plt
from numpy.linalg import norm

# Code is based on Advanced Lane Finding lesson material and walkthrough

class Vehicles():
	# A class that keeps track of vehicle tracks and assigns new detections to them
	# when starting a new instance specify all unassigned variables
	def __init__(self, MydistThresh = 25, MydrawThresh = 2, MyageKillThresh = 2):
		# list that stores all the past (left,right) center set values used for smoothing the output
		self.current_tracks = []

		# This is the farthest a new box can be from an existing track and still be assigned to the existing track,
		# otherwise a new track is created
		self.distThresh = MydistThresh

		# This is how many frames a track must be assigned detections before the plotter will plot it on the image
		# Increase this to reduce false positives at the cost of a time delay for new detections
		self.drawThresh = MydrawThresh

		# This is how many frames a track can go without being assigned new detections before it is deleted
		self.ageKillThresh = MyageKillThresh

		# Number of frames the tracker has seen (incremented when self.find_window_centroids() is called)
		self.frames_analyzed = 0

	# if raw_or_avgd is raw it returns a list of num_history most recent bounding boxes for each track
	# if raw_or_avgd is avgd it returns the average of the num_history most recent bounding boxes for each track
	def access_boxes(self, num_history, raw_or_avgd = 'avgd'):
		current_tracks = self.current_tracks
		# For each track, extract the boxes corresponding to the last num_history frames
		#aliveTracks = [track[3] for track in current_tracks if track[0] == 'alive']
		aliveTracks = [track[3] for track in current_tracks if ((track[0] == 'alive')|(track[0] == 'almostdead'))]

		if len(aliveTracks) > 0:
			if raw_or_avgd == 'raw':
				return [np.array(track[-min((len(track),num_history)):]).astype(int) for track in aliveTracks]
			else:
				return [np.mean(track[-min((len(track),num_history)):], axis=0).astype(int) for track in aliveTracks]
		else:
			print("No cars detected!")
			return []

	# find_window_centroids: The main tracking function for finding and storing lane segment positions
	# Input: a perspective transformed binary thresholded image
	# Method: Uses convolutions of a window template and sliding windows to determine the window position at each level
	#			moving up the image from the bottom.
	# Output: Adds the detected window locations and their counts to accumulator variables devined during initilization
	# Returns: averaged window locations, but this output isn't really used except for window plotting
	def assign_detections(self, new_boxes, num_history = 2, raw_or_avgd = 'avgd'):

		# Track objects are (status{"new","alive","almostdead","dead"}, numDetectionsInTrack, framesSinceLastSeen, {bbox age 1, bbox age 2, ...})
		# "dead" tracks are deleted from the list
		distThresh = self.distThresh
		drawThresh = self.drawThresh
		ageKillThresh = self.ageKillThresh

		current_tracks = self.current_tracks
		new_box_centers = [np.mean(np.array(box).astype(float), axis=0) for box in new_boxes]
		new_box_sizes = [np.diff(np.array(box).astype(float), axis=0)[0] for box in new_boxes]

		extendable_tracks = [track for track in current_tracks if track[0] != 'dead']

		# If we have any existing tracks, try to assign new detections to them
		if len(extendable_tracks) > 0:
			# Compute a cost matrix for use with the Hungarian/Munkres/Kuhn-Munkres assignment algorithm:
			# The cost matrix is the distance between the last box of each track (rows) and each new box (columns)
			# This is where you would implement an improvement using spline fits of the previous trajectories
			# 	the idea being if the track is already on its way to the new_box location, it doesn't matter how
			#	far away it WAS, it matters how much it had to "accelerate" between its last known location and speed
			#	to get to the proposed new_box location.
			cost = np.zeros((len(extendable_tracks), len(new_boxes)))

			# For each track
			for idx, track in enumerate(extendable_tracks):

				lastBox_center = np.mean(track[3][-1], axis=0).astype(float)
				lastBox_size = np.diff(track[3][-1], axis=0).astype(float)[0]

				# find the x distance between the two box centers divided by the smallest box width
				# do the same for the y distance. Compute the norm to get a distance metric.
				# can compute each row of the cost matrix by mapping over all new boxes
				cost[idx,:] = [norm(((lastBox_center[0]-new_box_c[0])/min(new_box_s[0], lastBox_size[0]),
					(lastBox_center[1]-new_box_c[1])/min(new_box_s[1], lastBox_size[1])))**2 for new_box_c, new_box_s in zip(new_box_centers, new_box_sizes)]

			assignments = linear_sum_assignment(cost)

			# First, take care of tracks that got assigned new_boxes
			for track_idx, box_idx in zip(assignments[0], assignments[1]):

				# If the distance to the nearest detected/matched new box is less than the threshold
				if cost[track_idx, box_idx] < distThresh:
					# Assign the new detection to the track

					extendable_tracks[track_idx][1] += 1 # add 1 to numDetectionsInTrack
					if extendable_tracks[track_idx][1] >= drawThresh:
						extendable_tracks[track_idx][0] = 'alive' # if the track has been around a while, label the track as 'alive' so it will get drawn
					extendable_tracks[track_idx][2] = 0 # reset framesSinceLastSeen to 0
					extendable_tracks[track_idx][3].append(new_boxes[box_idx]) # add the new box

				# If the distance to the matched box is too great and the track has not been assigned a detection for a while, kill/delete it
				elif extendable_tracks[track_idx][2] > ageKillThresh:
					extendable_tracks[track_idx][2] += 1 # add 1 to framesSinceLastSeen
					extendable_tracks[track_idx][1] += 1 # add 1 to numDetectionsInTrack
					extendable_tracks[track_idx][0] = 'dead'
				# Otherwise, mark it as getting old (which will stop plotting, but not delete it)
				else:
					extendable_tracks[track_idx][2] += 1 # add 1 to framesSinceLastSeen
					extendable_tracks[track_idx][1] += 1 # add 1 to numDetectionsInTrack
					extendable_tracks[track_idx][0] += 'almostdead'

			# Now take care of tracks that didn't get assigned new boxes
			for track_idx, track in enumerate(extendable_tracks):
				if track_idx not in assignments[0]:
					# If the track didn't get assigned this time and has not been assigned a detection for a while, kill/delete it
					if extendable_tracks[track_idx][2] > ageKillThresh:
						extendable_tracks[track_idx][2] += 1 # add 1 to framesSinceLastSeen
						extendable_tracks[track_idx][1] += 1 # add 1 to numDetectionsInTrack
						extendable_tracks[track_idx][0] = 'dead'
					# Otherwise, mark it as getting old (which will stop plotting, but not delete it)
					else:
						extendable_tracks[track_idx][2] += 1 # add 1 to framesSinceLastSeen
						extendable_tracks[track_idx][1] += 1 # add 1 to numDetectionsInTrack
						extendable_tracks[track_idx][0] += 'almostdead'

			# Now take care of detections that didn't get assigned to existing tracks by creating new tracks for them
			for box_idx, box in enumerate(new_boxes):
				if box_idx not in assignments[1]:
					new_track = ["new", 1, 0, [box]]
					extendable_tracks.append(new_track)

		else:
			for box_idx, box in enumerate(new_boxes):
				new_track = ["new", 1, 0, [box]]
				extendable_tracks.append(new_track)
		
		# Now set self.current_tracks to our new tracks!
		self.current_tracks = extendable_tracks

		# increase the frame counter for writing frame number overlay
		self.frames_analyzed += 1

		# returns the most recent boxes for each track
		return self.access_boxes(num_history=num_history, raw_or_avgd=raw_or_avgd)