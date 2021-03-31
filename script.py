import pandas as pd
import numpy as np
from math import log, pi, sqrt, exp
from multiprocessing import Pool
import matplotlib.pyplot as plt

#Class the holds priors for each pose
#Also holds the normal distributions for each attribute
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
		
	def calculate_likelihood(self, instance, mode, parameters):
		likelihood = 0 if self.prior == 0 else log(self.prior)
		instance = pd.to_numeric(instance).to_numpy()
		
		if mode == "classic":
			log_pdfs = -np.log(self.stdevs*sqrt(2*pi))-0.5*(((instance-self.means)/self.stdevs)**2)
			likelihood += np.sum(np.nan_to_num(log_pdfs, nan=0))
		
		if mode == "box_and_closest":
			instance = calculate_height_and_width(instance)
			log_pdfs = -np.log(self.stdevs*sqrt(2*pi))-0.5*(((instance-self.means)/self.stdevs)**2)
			likelihood += np.sum(np.nan_to_num(log_pdfs, nan=0))

			closest_points = calculate_closest_points(instance)
			closest_points_indexes = np.where(closest_points != -1)
			conditional_probs = self.closest_point_probs[closest_points_indexes, closest_points[closest_points_indexes]]
			likelihood += np.sum(np.log(conditional_probs))
					
		if mode == "KDE":
			bandwidth = parameters[0]
			likelihoods = (1/(bandwidth*sqrt(2*pi))) * np.exp(-0.5*(((self.data - instance)/bandwidth)**2))
			likelihoods = np.sum(np.nan_to_num(likelihoods, nan=0), axis = 0)
			likelihoods[likelihoods == 0] = np.nan
			likelihood += np.sum(np.nan_to_num(np.log(likelihoods), nan=0))
			
		if mode == "mean_imputation":
			instance[np.isnan(instance)] = self.means[np.isnan(instance)]
			log_pdfs = -np.log(self.stdevs*sqrt(2*pi))-0.5*(((instance-self.means)/self.stdevs)**2)
			likelihood += np.sum(np.nan_to_num(log_pdfs, nan=0))

		if mode == "absence_variable":
			log_pdfs = -np.log(self.stdevs*sqrt(2*pi))-0.5*(((instance-self.means)/self.stdevs)**2)
			likelihood += np.sum(np.nan_to_num(log_pdfs, nan=0))
			
			coordinate_present = np.isnan(instance[11:])
			
			absence_probs = self.absence_probs[coordinate_present]
			absence_probs[absence_probs == 0] = np.nan
			likelihood += np.sum(np.nan_to_num(np.log(absence_probs), nan=0))
			
			presence_probs = 1-self.absence_probs[np.logical_not(coordinate_present)]
			presence_probs[presence_probs == 0] = np.nan
			likelihood += np.sum(np.nan_to_num(np.log(presence_probs), nan=0))

		return likelihood

#Preprocessing: converts 9999 to np.NaN
def preprocess(filename):
	train = pd.read_csv(filename, header = None)
	train.replace(9999, np.NaN, inplace = True)
	return train
	
#Calculate the height and width of the pose
def calculate_height_and_width(instance):
	return np.array(max(instance[:11])-min(instance[:11]), max(instance[11:])-min(instance[11:]))
	
#Take an instance and return a list containing the closest point to every point, that is not nan
def calculate_closest_points(instance):
	points = np.dstack((instance[:11], instance[11:]))[0]
	#Distances is a 2D array the contains the distances between all points
	distances = np.array([np.sqrt(np.sum((point - points)**2, axis=1)) for point in points])
	#Assuming that no two body points share the same coordinates
	distances[distances == 0] = np.nan
	distances = np.nan_to_num(distances, nan=np.infty)
	closest_points = np.argmin(distances, axis = 0)
	closest_points_distances = np.min(distances, axis = 0)
	closest_points[closest_points_distances == np.infty] = -1
	return closest_points

#Calculate priors and attribute distributions for a given dataframe
#This dataframe should only hold data for a single class
#CHANGE THE GROUP TO A NUMPY ARRAY, 
def calculate_model_info(group, num_instances, mode):
	pose = Pose(group[0].iloc[0])
	group = group.iloc[:,1:]
	pose.prior = len(group)/num_instances
	if (mode == "classic" or mode == "mean_imputation"):
		pose.means = group.mean().to_numpy()
		pose.stdevs = group.std().to_numpy()
	if (mode == "KDE"):
		pose.data = group.to_numpy()
	if (mode == "absence_variable"):
		pose.means = group.mean().to_numpy()
		pose.stdevs = group.std().to_numpy()
		pose.absence_probs = (len(group) - group.iloc[:,11:].count().to_numpy())/len(group)
	if (mode == "box_and_closest"):
		widths_and_heights = pd.DataFrame([calculate_height_and_width(row[1][1:]) for row in group.iterrows()])
		pose.means = widths_and_heights.mean()
		pose.stdevs = widths_and_heights.std()
		calculate_closest_points(group)
		#Create a dataframe that has 11 columns, populated with the index of the closest points
		#This should be done with iterrows, list comprehsnsion and converting to a dataframe like line 116
		#On each row apply the calcualte_closest_points(row) to get the closest points
		#With this data frame, get the probability of each point being the closest point and store this in pose.closest_point_probs
	return pose

#Training: Determining priors and attribute distributions for every class
#Returns a pandas series that contains pose objects for every pose
#Each object contains priors and attribute distributions
def train(data, mode):
	poses = [calculate_model_info(data.loc[group[1].index], len(data), mode) for group in data.groupby([0])]
	#groups = data.groupby([0])
	#poses = groups.apply(calculate_model_info, num_instances=len(data), mode = mode)
	return poses

#Returns the name of the most likely post for any given instance
def predict_instance(instance, poses, mode, parameters):
	#Accounts for unpacking iterrows vs df.apply
	if len(instance) == 2:
		instance = instance[1]
	likelihoods = [pose.calculate_likelihood(instance[1:], mode, parameters) for pose in poses]
	return poses[np.argmax(likelihoods)].name

#Predicts the class labels for a dataframe
#Set speedup to true to use multiprocessing, false to use df.apply
#Multiprocessing will not work on windows or jupyter environments.
def predict(data, poses, mode, parameters, speedup):
	if speedup:
		pool_input = zip(data.iterrows(), [poses]*len(data), [mode]*len(data), [parameters]*len(data))
		with Pool(8) as pool:
			predictions = pool.starmap(predict_instance, pool_input)
	else:
		predictions = data.apply(predict_instance, poses = poses, mode = mode, parameters = parameters, axis = 1)
	return predictions

#Calculate accuracy of predictions
def evaluate(predictions, test):
	correct = sum(predictions==test[0])
	return 100*correct/len(predictions)
	
#Random hold out
def random_hold_out(data, hold_out_percent, mode, parameters):
	train_data = data.sample(frac = hold_out_percent, random_state = 3)
	test_data = data.drop(train_data.index)
	poses = train(train_data, mode)
	predictions = predict(test_data, poses, mode, parameters, True)
	print(evaluate(predictions, test_data))
	
#Cross validation
def cross_validation(data, num_partitions, mode, parameters):
	if num_partitions == -1:
		num_partitions = len(data)
	indexes = np.array(data.index)
	np.random.seed(3)
	np.random.shuffle(indexes)
	accuracy = 0
	for test_set_indexes in np.array_split(indexes, num_partitions):
		test_data = data.loc[test_set_indexes]
		train_data = data.drop(test_data.index)
		poses = train(train_data, mode)
		predictions = predict(test_data, poses, mode, parameters, True)
		accuracy += evaluate(predictions, test_data)
	return accuracy/num_partitions
	
#Graph for picking bandwidth parameter
def maximise_bandwidth(data, num_partitions, min_bandwidth, max_bandwidth, step):
	accuracies = []
	bandwidths = np.arange(min_bandwidth, max_bandwidth+step, step)
	for bandwidth in bandwidths:
		accuracy = cross_validation(data, num_partitions, "KDE", [bandwidth])
		print(bandwidth, accuracy)
		accuracies.append(accuracy)
	
	plt.plot(accuracies)
	plt.show()
	
	return bandwidths[np.argmax(accuracies)]
