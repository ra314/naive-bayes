import pandas as pd
import numpy as np
from math import log, pi, sqrt, exp
from multiprocessing import Pool
import matplotlib.pyplot as plt

#Class that holds:
#Priors for each pose and each attributes' normal distributions for the respective pose.
#The various likelihood functions used in training and prediction phases.
class Pose:
	def __init__(self, name):
		self.name = name
		self.prior = None
		self.means = []
		self.stdevs = []
		self.absence_probs = []
		self.closest_point_probs = []
		self.data = None
		
	def __str__(self):
		return f"Name: {self.name}, Prior: {self.prior}, Absence Probs: {self.absence_probs}"
		
	#Calculates the pdfs for a vector of means, stdevs and x values.
	#Then logs the pdfs and returns the sum.
	def log_pdf_sum(self, instance):
		log_pdfs = -np.log(self.stdevs*sqrt(2*pi))-0.5*(((instance-self.means)/self.stdevs)**2)
		return np.sum(np.nan_to_num(log_pdfs, nan=0))
	
	def calculate_likelihood(self, instance, mode, parameters):
		likelihood = 0 if self.prior == 0 else log(self.prior)
		instance = pd.to_numeric(instance).to_numpy()
		
		if mode == "classic":
			#Gaussian Naive Bayes
			likelihood += self.log_pdf_sum(instance)
		
		if mode == "box_and_closest":
			#Gaussian Naive Bayes on the height and width of a pose.
			pose_dims = calculate_height_and_width(instance)
			likelihood += self.log_pdf_sum(pose_dims)

			#For each point in an instance, get the index of the closest point to it.
			closest_points = calculate_closest_points(instance)
			#Naive Bayes using the closest point of every point as categorical attributes.
			closest_point_probs = self.closest_point_probs[np.where(closest_points != -1),[closest_points[closest_points!=-1]]]
			#If the probability of the closest point is 0, change it to np.nan, so that np.log can be applied.
			closest_point_probs[closest_point_probs == 0] = np.nan
			likelihood += np.sum(np.nan_to_num(np.log(closest_point_probs), nan=0))

		if mode == "KDE":
			#self.data contains all labelled instances for this pose.
			#The pdfs are calculated treating each data point from self.data
			#as the centre of a normal distribution with standard deviation = bandwidth.
			#For any given instance there will be 22*(len(self.data)) pdfs calculated.
			#These pdfs are then summed for each attribute. The 0's are converted to np.nan to allow np.log to be applied.
			bandwidth = parameters[0]
			pdfs = (1/(bandwidth*sqrt(2*pi))) * np.exp(-0.5*(((self.data - instance)/bandwidth)**2))
			sum_pdfs = np.sum(np.nan_to_num(pdfs, nan=0), axis = 0)
			sum_pdfs[sum_pdfs == 0] = np.nan
			likelihood += np.sum(np.nan_to_num(np.log(sum_pdfs), nan=0))
			
		if mode == "mean_imputation":
			#Impute missing points with the pose's average value for that point, then perform Gaussian Naive Bayes.
			instance[np.isnan(instance)] = self.means[np.isnan(instance)]
			likelihood += self.log_pdf_sum(instance)

		if mode == "absence_variable":
			#Regular Guassian Naive Bayes.
			likelihood += self.log_pdf_sum(instance)
			
			#This is a boolean array. True if the coordinate pair is missing, False otherwise.
			coordinates_absent = np.isnan(instance[11:])
			
			#Naive Bayes is applied using the absence or presence of every coordinate pair as a categorical attribute.
			absence_probs = self.absence_probs[coordinates_absent]
			absence_probs[absence_probs == 0] = np.nan
			likelihood += np.sum(np.nan_to_num(np.log(absence_probs), nan=0))
			
			presence_probs = 1-self.absence_probs[np.logical_not(coordinates_absent)]
			presence_probs[presence_probs == 0] = np.nan
			likelihood += np.sum(np.nan_to_num(np.log(presence_probs), nan=0))

		return likelihood

#Preprocessing: converts 9999 values to np.NaN.
def preprocess(filename):
	train = pd.read_csv(filename, header = None)
	train.replace(9999, np.NaN, inplace = True)
	return train
	
#Convert instance (1x22) into coordinates (11x2).
def get_coordinates(instance):
	return np.dstack((instance[:11], instance[11:]))[0]
	
#Calculate the height and width of the instance.
def calculate_height_and_width(instance):
	#Convert the instance into coordinates. Convert it into floats, keep the non np.nan values.
	#Split the array in two parts, x coordinates and y coordinates.
	instance = get_coordinates(instance)
	instance = np.split(instance[np.logical_not(np.isnan(instance.astype(np.float)))], 2)
	if len(instance[0]) == 0:
		return np.array([0,0])
	return np.array([max(instance[0])-min(instance[0]), max(instance[1])-min(instance[1])])

#Take an instance and return a list containing the closest point to every point, that is not nan.
def calculate_closest_points(instance):
	points = get_coordinates(instance)
	#Distances is a 2D array the contains the distances between all points
	distances = np.array([np.sqrt(np.sum((point - points)**2, axis=1)) for point in points])
	#Assuming that no two body points share the same coordinates.
	#Set distance from the point to itself, and the distance for missing points as infinity.
	distances[distances == 0] = np.nan
	distances = np.nan_to_num(distances, nan=np.infty)
	closest_points = np.argmin(distances, axis = 0)
	closest_points_distances = np.min(distances, axis = 0)
	closest_points[closest_points_distances == np.infty] = -1
	return closest_points

#Calculate priors and attribute distributions for a given dataframe depending on the mode provided.
#This dataframe should only hold data for a single class.
def calculate_model_info(group, num_instances, mode):
	pose = Pose(group[0].iloc[0])
	group = group.iloc[:,1:]
	pose.prior = len(group)/num_instances
	if (mode == "classic" or mode == "mean_imputation" or mode == "absence_variable"):
		#Find mean and stdev of the coordinates for Gaussian Naive Bayes.
		pose.means = group.mean().to_numpy()
		pose.stdevs = group.std().to_numpy()
		
		if (mode == "absence_variable"):
			#Find probability of each point being absent.
			pose.absence_probs = (len(group) - group.iloc[:,11:].count().to_numpy())/len(group)
			
	if (mode == "KDE"):
		#Storing the group data as a numpy array in the Pose object.
		pose.data = group.to_numpy()
		
	if (mode == "box_and_closest"):
		#Find the mean and stdev of the height and width of every instance for Gaussian Naive Bayes.
		widths_and_heights = pd.DataFrame([calculate_height_and_width(row[1]) for row in group.iterrows()])
		pose.means = widths_and_heights.mean()
		pose.stdevs = widths_and_heights.std()
		
		#Find the closest point of every point in the data frame. 
		#The input is a (len(group)x22) data frame. The output is a (len(group)x11) data frame.
		closest_points = pd.DataFrame([calculate_closest_points(row[1]) for row in group.iterrows()])
		#Then the probability of each point being the closest point for each body point is calculated for categorical Naive Bayes.
		pose.closest_point_probs = np.zeros((11,12))
		for column_index in closest_points:
			counts = closest_points[column_index].value_counts()
			pose.closest_point_probs[column_index][counts.index] = counts.values
		pose.closest_point_probs = pose.closest_point_probs[:,:-1]
		pose.closest_point_probs = pose.closest_point_probs/(np.sum(pose.closest_point_probs, axis = 1)).reshape(11,1)
	return pose

#Training: Determines priors and attribute distributions for every class.
#Returns a pandas series that contains pose objects for every pose.
#Each object contains priors and attribute distributions.
def train(data, mode):
	poses = [calculate_model_info(data.loc[group[1].index], len(data), mode) for group in data.groupby([0])]
	return poses

#Returns the name of the most likely pose for any given instance.
def predict_instance(instance, poses, mode, parameters):
	#Accounts for unpacking iterrows vs df.apply
	if len(instance) == 2:
		instance = instance[1]
	likelihoods = [pose.calculate_likelihood(instance[1:], mode, parameters) for pose in poses]
	return poses[np.argmax(likelihoods)].name

#Predicts the class labels for a dataframe.
#Set speedup to true to use multiprocessing, false to use df.apply.
#Multiprocessing will not work on windows or jupyter environments.
def predict(data, poses, mode, parameters, speedup):
	if speedup:
		pool_input = zip(data.iterrows(), [poses]*len(data), [mode]*len(data), [parameters]*len(data))
		with Pool(8) as pool:
			predictions = pool.starmap(predict_instance, pool_input)
	else:
		predictions = data.apply(predict_instance, poses = poses, mode = mode, parameters = parameters, axis = 1)
	return predictions

#Calculate accuracy of predictions.
def evaluate(predictions, test):
	correct = sum(predictions==test[0])
	return 100*correct/len(predictions)
	
#Random hold out.
def random_hold_out(data, hold_out_percent, mode, parameters):
	train_data = data.sample(frac = hold_out_percent, random_state = 3)
	test_data = data.drop(train_data.index)
	poses = train(train_data, mode)
	predictions = predict(test_data, poses, mode, parameters, True)
	print(evaluate(predictions, test_data))
	
#Cross validation.
def cross_validation(data, num_partitions, mode, parameters):
	#If num_paritions is set to -1, perform leave one out cross validation.
	if num_partitions == -1:
		num_partitions = len(data)
	#We take the indexes of the data, and shuffle them.
	indexes = np.array(data.index)
	np.random.seed(3)
	np.random.shuffle(indexes)
	accuracy = 0
	#Perform train, test and evaluate on each partition.
	for test_set_indexes in np.array_split(indexes, num_partitions):
		test_data = data.loc[test_set_indexes]
		train_data = data.drop(test_data.index)
		poses = train(train_data, mode)
		predictions = predict(test_data, poses, mode, parameters, True)
		accuracy += evaluate(predictions, test_data)
	return accuracy/num_partitions
	
#Graph displaying the accuracy of KDE at a variety of bandwidths.
#The function returns the bandwidth that maximised accuracy.
def optimize_bandwidth(data, num_partitions, min_bandwidth, max_bandwidth, step):
	accuracies = []
	bandwidths = np.arange(min_bandwidth, max_bandwidth+step, step)
	for bandwidth in bandwidths:
		accuracy = cross_validation(data, num_partitions, "KDE", [bandwidth])
		accuracies.append(accuracy)
	
	plt.plot(accuracies)
	plt.show()
	
	return bandwidths[np.argmax(accuracies)]
	
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
		
	plt.show()
