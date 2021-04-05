import numpy as np
	
#Convert instance (1x22) into coordinates (11x2).
def get_coordinates(instance):
	return np.dstack((instance[:11], instance[11:]))[0]
	
#Binning width to height ratio
def bin_height_to_width_ratio(height_to_width_ratio):
	#Tall
	if (height_to_width_ratio<0.75):
		return 0
	#Medium
	elif (height_to_width_ratio<1.5):
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
		return height_and_width[0]/height_and_width[1]
	elif mode == "discretized_height_to_width_ratio":
		return bin_height_to_width_ratio(height_and_width[0]/height_and_width[1])

#Take an instance and return a list containing the closest point to every point, that is not nan.
def calculate_closest_points(instance):
	points = get_coordinates(instance)
	#Distances is a 2D array the contains the distances between all points.
	distances = np.array([np.sqrt(np.sum((point - points)**2, axis=1)) for point in points])
	#Assuming that no two body points share the same coordinates.
	#Set distance from the point to itself, and the distance for missing points as infinity.
	distances[distances == 0] = np.infty
	distances = np.nan_to_num(distances, nan=np.infty)
	closest_points = np.argmin(distances, axis = 0)
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