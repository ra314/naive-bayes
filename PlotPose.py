import pandas as pd
import matplotlib.pyplot as plt
	
#Connect two points on a plot.
def connect_points(point1, point2):
	plt.plot([point1[0], point2[0]], [point1[1], point2[1]])	
	
#Plotting poses.
def plot_pose(instance):
	plt.title(instance[0])
	points = get_coordinates(instance[1:])
	#Added a dummy point for easier indexing
	points = np.concatenate([[[np.nan, np.nan]], points])
	plt.scatter(points[:,0], points[:,1])
	
	#Annotating the points.
	for i in range(1,12):
		if points[i].all():
			plt.annotate(i, points[i])
	
	#Drawing lines between body points.
	if points[1].all() and points[2].all():
		connect_points(points[1], points[2])
	if points[2].all() and points[3].all():
		connect_points(points[2], points[3])
	if points[3].all() and points[4].all():
		connect_points(points[3], points[4])
	if points[2].all() and points[5].all():
		connect_points(points[2], points[5])
	if points[5].all() and points[6].all():
		connect_points(points[5], points[6])
	if points[2].all() and points[7].all():
		connect_points(points[2], points[7])
	if points[7].all() and points[8].all():
		connect_points(points[7], points[8])
	if points[8].all() and points[9].all():
		connect_points(points[8], points[9])
	if points[7].all() and points[10].all():
		connect_points(points[7], points[10])
	if points[10].all() and points[11].all():
		connect_points(points[10], points[11])
		
	plt.show(block=False)
	
def plot_all_poses(data, seconds_between_poses):
	plt.ion()
	for row in data.iterrows():
		plot_pose(row[1])
		plt.pause(seconds_between_poses)
		plt.clf()
	plt.close()
	
#Plotting widths and heights of poses.
def plot_heights_and_widths(data):
	widths_and_heights = pd.DataFrame([calculate_height_and_width(row[1][1:]) for row in data.iterrows()])
	for group in data.groupby([0]):
		group_widths_and_heights = widths_and_heights.loc[group[1].index]
		plt.scatter(group_widths_and_heights[0], group_widths_and_heights[1], label=group[0])
	plt.show()
