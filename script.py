import pandas as pd
import numpy as np
from math import log, pi, sqrt, exp
from multiprocessing import Pool

#Takes log, but returns 0, if the value is 0
def log_0(x):
	if x == 0:
		return 0
	else:
		return log(x)

#Takes a mean, variance and x value and returns the log of the density
#Removed (sqrt(2*pi)) from the calcuation since it's a constant
def pdf(normal, x, mode):
	mean = normal.mean
	stdev = normal.stdev
	if mode == "classic":
		density = (1/(stdev*sqrt(2*pi))) * exp((-1/2)*(((x-mean)/stdev)**2))
	if mode == "log":
		density = -log_0(stdev*sqrt(2*pi)) -(0.5)*(((x-mean)/stdev)**2)
	return density

#Class the holds priors for each pose
#Also holds the normal distributions for each attribute
class Pose:
	def __init__(self, name):
		self.name = name
		self.prior = None
		self.normals = []
		self.absence_probs = []
		self.closest_point_probs = []
		self.data = None
		
	def __str__(self):
		return f"Name: {self.name}, Prior: {self.prior}, Absence Probs: {self.absence_probs}"
		
	def calculate_likelihood(self, instance, mode, parameters):
		likelihood = log_0(self.prior)
		if mode == "classic":
			for normal, attribute in zip(self.normals, instance):
				if not(np.isnan(attribute)):
					likelihood += pdf(normal, attribute, "log")
		
		if mode == "box_and_closest":
			for normal, attribute in zip(self.normals, calculate_height_and_width(instance)):
				if not(np.isnan(attribute)):
					likelihood += pdf(normal, attribute, "log")
					
		if mode == "KDE":
			bandwidth = parameters[0]
			for attribute, column_index in zip(instance, data.iloc[:,1:]):
				if not(np.isnan(attribute)):
					total_pdf = 0
					for value in self.data[column_index]:
						if not(np.isnan(value)):
							total_pdf += pdf(Normal(value, bandwidth), attribute, "classic")
					likelihood += log_0(total_pdf/len(data))

		if mode == "mean_imputation":
			for normal, attribute in zip(self.normals, instance):
				if np.isnan(attribute):
					attribute = normal.mean
				likelihood += pdf(normal, attribute, "log")

		if mode == "absence_variable":
			for normal, absence_prob, attribute in zip(self.normals, self.absence_probs, instance):
				if not(np.isnan(attribute)):
					likelihood += pdf(normal, attribute, "log")
					likelihood += log_0(1-absence_prob)
				else:
					likelihood += log_0(absence_prob)

		return likelihood

#Class that decribes a normal distribution with a certain mean and standard deviation	
class Normal:
	def __init__(self, mean, stdev):
		self.mean = mean
		self.stdev = stdev
		
	def __str__(self):
		return f"Mean: {self.mean}, Standard Deviation: {self.stdev}"

#Preprocessing: converts 9999 to np.NaN
def preprocess(filename):
	train = pd.read_csv(filename, header = None)
	train.replace(9999, np.NaN, inplace = True)
	return train
	
#Calculate the height and width of the pose
def calculate_height_and_width(instance):
	return [max(instance[:11])-min(instance[:11]), max(instance[11:])-min(instance[11:])]
	
#Calculates distance between two points
def calculate_distance(x1, y1, x2, y2):
	return sqrt((x1 - x2)**2 + (y1 - y2)**2)
	
#Take an instance and return a list containing the closest point to every point, that is not nan
def calculate_closest_points(instance):
	points = [[instance.iloc[0, i], instance.iloc[0, i+11]] for i in range(len(instance - 1)/2)]
	neighbours = []
	for i in range(len(points)):
		closest = -1
		mindist = np.inf
		for j in range(len(points)):
			if i != j:
				dist = calculate_distance(points[i][0], points[i][1], points[j][0], points[j][1])
				if dist < mindist:
					mindist = dist
					closest = j
		neighbours.append(j)
	#print(points)

#Calculate priors and attribute distributions for a given dataframe
#This dataframe should only hold data for a single class
def calculate_model_info(group, num_instances, mode):
	pose = Pose(group[0].iloc[0])
	pose.prior = len(group[1])/num_instances
	if (mode == "classic" or mode == "mean_imputation"):
		pose.normals= [Normal(mean, stdev) for mean, stdev in zip(group.mean(), group.std())]
	if (mode == "KDE"):
		pose.data = group.copy()
	if (mode == "absence_variable"):
		pose.normals= [Normal(mean, stdev) for mean, stdev in zip(group.mean(), group.std())]
		pose.absence_probs = (len(group) - group.count())/len(group)
	if (mode == "box_and_closest"):
		widths_and_heights = pd.DataFrame([calculate_height_and_width(row[1][1:]) for row in group.iterrows()])
		pose.normals = [Normal(mean, stdev) for mean, stdev in zip(widths_and_heights.mean(), widths_and_heights.std())]
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
	groups = data.groupby([0])
	poses = groups.apply(calculate_model_info, num_instances=len(data), mode = mode)
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
	
