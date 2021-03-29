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
		self.data = None
		
	def __str__(self):
		return f"Name: {self.name}, Prior: {self.prior}"
		
	def calculate_likelihood(self, instance, mode, parameters):
		likelihood = 0
		if mode == "classic":
			for normal, attribute in zip(self.normals, instance):
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

#Calculate priors and attribute distributions for a given dataframe
#This dataframe should only hold data for a single class
def calculate_model_info(group, num_instances, mode):
	pose = Pose(group[0].iloc[0])
	pose.prior = len(group[1])/num_instances
	if (mode == "classic"):
		for mean, stdev in zip(group.mean(), group.std()):
			pose.normals.append(Normal(mean, stdev))
	if (mode == "KDE"):
		pose.data = data.loc[data[0] == pose.name]
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
