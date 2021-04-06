import numpy as np
from itertools import combinations
	
#Convert instance (1x22) into coordinates (11x2).
def get_coordinates(instance):
	return np.dstack((instance[:11], instance[11:]))[0]
	
#Binning width to height ratio
def bin_height_to_width_ratio(height_to_width_ratio):
	#Tall
	if (height_to_width_ratio>4/3):
		return 0
	#Medium
	elif (height_to_width_ratio>2/3):
		return 1
	#Wide
	else:
		return 2
	
#Calculate the height and width of the instance.
def calculate_height_and_width(instance, mode):
	#Convert the instance into coordinates. Convert it into floats, keep the non np.nan values.
	#Split the array in two parts, x coordinates and y coordinates.
	instance = get_coordinates(instance)
	instance = np.split(instance[np.logical_not(np.isnan(instance.astype(np.float)))], 2)
	if len(instance[0]) == 0:
		if mode == "height_and_width":
			return np.array([np.nan,np.nan])
		else:
			return np.nan
	
	height_and_width = np.array([max(instance[0])-min(instance[0]), max(instance[1])-min(instance[1])])
	
	if mode == "height_and_width":
		return height_and_width
	elif mode == "height_to_width_ratio":
		return height_and_width[1]/height_and_width[0]
	elif mode == "discretized_height_to_width_ratio":
		return bin_height_to_width_ratio(height_and_width[1]/height_and_width[0])

#Take an instance and return a list containing the nth closest point to every point, that is not nan.
def calculate_closest_points(instance, n):
	points = get_coordinates(instance)
	#Distances is a 2D array the contains the distances between all points.
	distances = np.array([np.sqrt(np.sum((point - points)**2, axis=1)) for point in points])
	#Assuming that no two body points share the same coordinates.
	#Set distance from the point to itself, and the distance for missing points as infinity.
	distances[distances == 0] = np.infty
	distances = np.nan_to_num(distances, nan=np.infty)
	closest_points = np.argsort(distances, axis = 0)[n]
	closest_points_distances = np.min(distances, axis = 0)
	closest_points[closest_points_distances == np.infty] = -1
	return closest_points

#Calculating the number of arms above the head.
def calculate_num_arms_above_head(instance):
	num = 0
	r_arm_y = [instance[3+11], instance[4+11]]
	l_arm_y = [instance[5+11], instance[6+11]]
	head_y = instance[1]
	
	if not np.isnan(r_arm_y).all():
		num += np.nanmean(r_arm_y) > head_y
	
	if not np.isnan(l_arm_y).all():
		num += np.nanmean(l_arm_y) > head_y
		
	if np.isnan(head_y):
		return np.nan
	else:
		return num
		
#Determine if the torso is perpendicular to the ground
def calculate_perpendicular_torso(instance):
	head_x = instance[1]
	neck_x = instance[2]
	hip_x = instance[7]
	alignment_diff = np.nansum(np.abs([head_x-neck_x, neck_x-hip_x, hip_x-head_x]), axis=0)
	num_present_points = 3-np.sum(np.isnan([head_x, neck_x, hip_x]))
	
	if num_present_points <= 1:
		return np.nan
	
	#Dividing the differences in alignment by the number of relevant body points present
	#This way having more body points doesn't make being aligned harder
	alignment_diff /= (num_present_points-1)
	
	return alignment_diff < 100
	
#Calculate the distance between each point
def calculate_distance_between_points(instance):
	points = get_coordinates(instance)
	point_combos = np.array(list(combinations(points, 2)))
	distances = np.sqrt(np.sum((point_combos[:,0]-point_combos[:,1])**2, axis=1))
	return distances
	
#Calculate the angle between two vectors:
def calculate_angle(line1, line2):
	return np.degrees(np.arccos(np.dot(line1, line2)/(np.linalg.norm(line1)*np.linalg.norm(line2))))

#Calculating the key angles
def calculate_key_angles(instance):
	#Labelling and getting all the points
	points = np.concatenate((np.zeros((1,2)), get_coordinates(instance)))
	head = points[1]
	neck = points[2]
	r_elbow = points[3]
	r_hand = points[4]
	l_elbow = points[5]
	l_hand = points[6]
	hip = points[7]
	r_knee = points[8]
	r_foot = points[9]
	l_knee = points[10]
	l_foot = points[11]
	
	#Calculating the import angles
	angles = np.zeros(7)
	angles[0] = calculate_angle(r_knee-hip, l_knee-hip)
	
	angles[1] = calculate_angle(r_hand-r_elbow, neck-r_elbow)
	angles[2] = calculate_angle(l_hand-l_elbow, neck-l_elbow)
	
	angles[3] = calculate_angle(r_elbow-neck, hip-neck)
	angles[4] = calculate_angle(l_elbow-neck, hip-neck)
	
	angles[5] = calculate_angle(r_foot-r_knee, hip-r_knee)
	angles[6] = calculate_angle(l_foot-l_knee, hip-l_knee)
	
	return angles
