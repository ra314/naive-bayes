import pandas as pd
import numpy as np
from math import log

#Takes a mean, variance and x value and returns the log of the density
#Removed (sqrt(2*pi)) from the calcuation since it's a constant
def pdf(normal, x):
	mean = normal.mean
	stdev = normal.stdev
	relative_sum = - log(stdev) - (1/2)*(((x-mean)/stdev)**2)
	return relative_sum

#Class the holds priors for each pose
#Also holds the normal distributions for each attribute
class Pose:
	def __init__(self, name):
		self.name = name
		self.prior = None
		self.normals = []
		
	def __str__(self):
		return f"Name: {self.name}, Prior: {self.prior}"
		
	def calculate_likelihood(self, instance):
		likelihood = 0
		for normal, value in zip(self.normals, instance):
			if not(np.isnan(value)):
				likelihood += pdf(normal, value)
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
def calculate_model_info(group, num_instances):
	pose = Pose(group[0].iloc[0])
	pose.prior = len(group[1])/num_instances
	for mean, stdev in zip(group.mean(), group.std()):
		pose.normals.append(Normal(mean, stdev))
	return pose

#Training: Determining priors and attribute distributions for every class
#Returns a pandas series that contains pose objects for every pose
#Each object contains priors and attribute distributions
def train(data):
	groups = data.groupby([0])
	poses = groups.apply(calculate_model_info, num_instances=len(data))
	return poses

#Returns the name of the most likely post for any given instance
def predict_instance(instance, poses):
	likelihoods = [pose.calculate_likelihood(instance[1:]) for pose in poses]
	return poses[np.argmax(likelihoods)].name

#Predicts the class labels for a dataframe
def predict(data, poses):
	predictions = data.apply(predict_instance, poses = poses, axis = 1)
	return predictions

#Calculate accuracy of predictions
def evaluate(predictions, test):
	correct = sum(predictions==test[0])
	return 100*correct/len(predictions)

'''
data = preprocess('train.csv')
poses = train(data)
instance = data.iloc[6]
poses['bridge'].calculate_likelihood(list(instance[1:]))
predictions = predict(data, poses)
evaluate(predictions, data)
'''
