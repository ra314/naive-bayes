import pandas as pd
import numpy as np
from math import log

#Create a function that takes a mean, variance and x values and returns the log density
def pdf(normal, x):
	mean = normal.mean
	stdev = normal.stdev
	relative_sum = log(1/stdev) + (-1/2)*(((x-mean)/stdev)**2)
	return relative_sum

class Pose:
	def __init__(self, name):
		self.name = name
		self.prior = None
		self.normals = []
		
	def __str__(self):
		return f"Name: {self.name}, Prior: {self.prior}"
		
	def calculate_likelihood(self, instance):
		likelihood = 1
		for normal, value in zip(self.normals, instance):
			if not(np.isnan(value)):
				likelihood *= pdf(normal, value)
		return likelihood
		
class Normal:
	def __init__(self, mean, stdev):
		self.mean = mean
		self.stdev = stdev
		
	def __str__(self):
		return f"Mean: {self.mean}, Standard Deviation: {self.stdev}"

#The function below shows that the highest non 9999 value in the datagrame is 1849
def sort_entire_df(df):
	all = df.stack().tolist()
	all = [item for item in all if not isinstance(item, str)]
	return sorted(all)

#Convert 9999 to Nan, or None
#Convert it into a data frame, which is the return value
def preprocess(filename):
	train = pd.read_csv(filename, header = None)
	train.replace(9999, np.NaN, inplace = True)
	return train

#Calculate priors and likelihoods for a give class of data
def calculate_model_info(group, num_instances):
	pose = Pose(group[0].iloc[0])
	pose.prior = len(group[1])/num_instances
	for mean, stdev in zip(group.mean(), group.std()):
		pose.normals.append(Normal(mean, stdev))
	return pose


#Training
#Groupby each class name
#Calculate the mean and variance through pandas
#Assign mean and variance to each attribute of each class
#Calcaulte the priors, by the size of groupby / total instance
#Return or modify a dictionary poses
def train(data):
	groups = data.groupby([0])
	poses = groups.apply(calculate_model_info, num_instances=len(data))
	return poses
	
def predict_instance(instance, poses):
	likelihoods = [pose.calculate_likelihood(instance[1:]) for pose in poses]
	return poses[np.argmax(likelihoods)].name

#Prediction takes a test dataset and return all predited class labels
#It calls the predict_instance function and applies using df.apply
#Computes that log probability for each class according to the formula on page 3 of the spec.
#Pick the class with the highest probability
def predict(data, poses):
	predictions = data.apply(predict_instance, poses = poses, axis = 1)
	return predictions

#Takes a test data set, uses predict function to get all the predicted classes
#Returns a percentage accuracy score
#Assumes labels are in first column of each dataframe
def evaluate(predictions, test):
	correct = sum(predictions[0]==test[0])
	return 100*correct/len(predictions[0])

'''
data = preprocess('train.csv')
poses = train(data)
instance = data.iloc[6]
poses['bridge'].calculate_likelihood(list(instance[1:]))
predictions = predict(data, poses)
'''
