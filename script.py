import pandas as pd
import numpy as np
from math import log

class Pose:
	def __init__(self, name):
		self.name = name
		self.prior = None
		self.likelihoods = []
		
	def __str__(self):
		printf("Name: {name}, Prior: {prior}")
		
class Likelihood:
	def __init__(self, mean, stdev):
		self.mean = mean
		self.stdev = stdev
		
	def __str__(self):
		print(f"Mean: {mean}, Standard Deviation: {stdev}")

Poses = {}

#The function below shows that the highest non 9999 value in the datagrame is 1849
def sort_entire_df(df):
	all = df.stack().tolist()
	all = [item for item in all if not isinstance(item, str)]
	return sorted(all)

#Convert 9999 to Nan, or None
#Convert it into a data frame, which is the return value
def preprocess(filename):
	train = pd.read_csv(filename, header = None)
	train.replace(9999, pd.NA, inplace = True)
	return train

#Calculate priors and likelihoods for a give class of data
def calculate_model_info(group):
	pose = Pose(group[0])
	pose.prior = len(group[1])/num_samples
	for mean, stdev in zip(group[1].mean(), group[1].std()):
		pose.likelihoods.append(Likelihood(mean, stdev))
	return pose

#Training
#Groupby each class name
#Calcaulte the mean and variance through pandas
#Assign mean and variance to each attribute of each class
#Calcaulte the priors, by the size of groupby / total instance
#Return or modify a dictionary poses
def train(data):
	groups = data.groupby([0])
	poses = groups.apply(calculate_model_info(len(data)))
	return poses

#Create a function that takes a mean, variance and x values and returns the log density
def pdf(likelihood, x):
	mean = likelihood.mean
	stdev = likelihood.stdev
	relative_sum = log(1/stdev) + (-1/2)*(((x-mean)/stdev)^2)
	return relative_sum


#Prediction takes a test dataset and return all predited class labels
#It calls the predict_instance function and applies using df.apply
#Computes that log probability for each class according to the formula on page 3 of the spec.
#Pick the class with the highest probability
def predict():
	return

#Takes a test data set, uses predict function to get all the predicted classes
#Returns a percentage accuracy score
def evaluate():
	return
