import numpy as np
from math import log, pi, sqrt, exp
import pandas as pd

from InstanceCalculations import calculate_height_and_width, calculate_closest_points, calculate_num_arms_above_head

#Class that holds:
#Priors for each pose and each attributes' normal distributions for the respective pose.
#The various likelihood functions used in training and prediction phases.
class Pose:
	def __init__(self, name):
		self.name = name
		self.prior = None
		
		self.coordinate_means = []
		self.coordinate_stdevs = []
		
		self.data = None
		
		self.absence_probs = []
		
		self.height_and_width_means = []
		self.height_and_width_stdevs = []
		
		self.height_to_width_ratio_means = []
		self.height_to_width_ratio_stdevs = []
		
		self.discretized_height_to_width_ratio_probs = []
		
		self.closest_point_probs = []
		
		self.arms_above_head_probs = []
		
	def __str__(self):
		return f"Name: {self.name}, Prior: {self.prior}, Absence Probs: {self.absence_probs}"
		
	#Calculates the pdfs for a vector of means, stdevs and x values.
	#Then logs the pdfs and returns the sum.
	def log_pdf_sum(self, instance, means, stdevs):
		log_pdfs = -np.log(stdevs*sqrt(2*pi))-0.5*(((instance-means)/stdevs)**2)
		return np.nansum(log_pdfs)
	
	def calculate_likelihood(self, instance, mode, parameters):
		likelihood = 0 if self.prior == 0 else log(self.prior)
		instance = pd.to_numeric(instance).to_numpy()
		
		#Gaussian Naive Bayes on coordinates of a pose.
		if "classic" in mode:
			likelihood += self.log_pdf_sum(instance, self.coordinate_means, self.coordinate_stdevs)
		
		#KDE Naive Bayes on coordinates of a pose.
		if "KDE" in mode:
			#self.data contains all labelled instances for this pose.
			#The pdfs are calculated treating each data point from self.data
			#as the centre of a normal distribution with standard deviation = bandwidth.
			#For any given instance there will be 22*(len(self.data)) pdfs calculated.
			#These pdfs are then summed for each attribute. 
			bandwidth = parameters[0]
			pdfs = (1/(bandwidth*sqrt(2*pi))) * np.exp(-0.5*(((self.data - instance)/bandwidth)**2))
			sum_pdfs = np.nansum(pdfs, axis = 0)
			sum_pdfs[sum_pdfs == 0] = np.nan
			likelihood += np.nansum(np.log(sum_pdfs))
		
		#Categorical Naive Bayes on the absence of the coordinates of a pose.
		if "coordinate_absence" in mode:			
			#This is a boolean array. True if the coordinate pair is missing, False otherwise.
			coordinates_absent = np.isnan(instance[11:])
			
			absence_probs = self.absence_probs[coordinates_absent]
			absence_probs[absence_probs == 0] = np.nan
			likelihood += np.nansum(np.log(absence_probs))
		
		#Categorical Naive Bayes on the presence of the coordiantes of a pose.
		if "coordinate_presence" in mode:			
			#This is a boolean array. True if the coordinate pair is missing, False otherwise.
			coordinates_absent = np.isnan(instance[11:])
				
			presence_probs = 1-self.absence_probs[np.logical_not(coordinates_absent)]
			presence_probs[presence_probs == 0] = np.nan
			likelihood += np.nansum(np.log(presence_probs))
		
		#Gaussian Naive Bayes on the height and width of a pose.
		if "height_and_width" in mode:
			height_and_width = calculate_height_and_width(instance, "height_and_width")
			likelihood += self.log_pdf_sum(height_and_width, self.height_and_width_means, self.height_and_width_stdevs)
		
		#Gaussian Naive Bayes on the height to width ratio of a pose.
		if "height_to_width_ratio" in mode:
			height_to_width_ratio = calculate_height_and_width(instance, "height_to_width_ratio")
			likelihood += self.log_pdf_sum(height_to_width_ratio, self.height_to_width_ratio_means, self.height_to_width_ratio_stdevs)
			
		#Categorical Naive Bayes on the discretized height to width ratio of a pose.
		if "discretized_height_to_width_ratio" in mode:
			discretized_height_to_width_ratio = calculate_height_and_width(instance, "discretized_height_to_width_ratio")
			likelihood += log(self.discretized_height_to_width_ratio_probs[int(discretized_height_to_width_ratio)])
			
		#Categorical Naive Bayes on the closest point of every point of a pose.
		if "closest_points" in mode:
			#For each point in an instance, get the index of the closest point to it.
			closest_points = calculate_closest_points(instance)
			
			closest_point_probs = self.closest_point_probs[np.where(closest_points != -1),[closest_points[closest_points!=-1]]]
			closest_point_probs[closest_point_probs == 0] = np.nan
			likelihood += np.nansum(np.log(closest_point_probs))
		
		#Categorical Naive Bayes on the number of arms above the head of a pose.
		if "arms_above_head" in mode:
			arms_above_head = calculate_num_arms_above_head(instance)
			if not np.isnan(arms_above_head):
				likelihood += log(self.arms_above_head_probs[arms_above_head])

		return likelihood