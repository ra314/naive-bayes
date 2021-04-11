import pandas as pd
import numpy as np
from multiprocessing import Pool

from PoseClass import Pose
from InstanceCalculations import calculate_height_and_width, calculate_closest_points, calculate_num_arms_above_head, calculate_perpendicular_torso, calculate_distance_between_points, calculate_key_angles
from PlotPose import *

#Preprocessing: converts 9999 values to np.NaN.
def preprocess(filename):
	train = pd.read_csv(filename, header = None)
	train.replace(9999, np.NaN, inplace = True)
	return train

#Calculate priors and attribute distributions for a given dataframe depending on the mode provided.
#This dataframe should only hold data for a single class.
def calculate_model_info(group, num_instances, mode, parameters):
	pose = Pose(group[0].iloc[0])
	pose.prior = len(group)/num_instances
	group = group.iloc[:,1:]
	
	if "classic" in mode:
		#Find mean and stdev of the coordinates for Gaussian Naive Bayes.
		pose.coordinate_means = group.mean().to_numpy()
		pose.coordinate_stdevs = group.std().to_numpy()
		
	if "KDE" in mode:
		#Storing the group data as a numpy array in the Pose object for KDE Naive Bayes.
		pose.data = group.to_numpy()
		
	if "coordinate_absence" in mode or "coordinate_presence" in mode:
		#Find probability of each point being absent for Categorical Naive Bayes.
		#Using Laplace add 1 smoothing.
		pose.absence_probs = ((len(group) - group.iloc[:,11:].count().to_numpy())+1)/len(group)
		
	if "height_and_width" in mode:
		#Find the mean and stdev of the height and width for Gaussian Naive Bayes.
		heights_and_widths = pd.DataFrame([calculate_height_and_width(row[1], "height_and_width") for row in group.iterrows()])
		pose.height_and_width_means = heights_and_widths.mean()
		pose.height_and_width_stdevs = heights_and_widths.std()
	
	if "height_to_width_ratio" in mode:
		#Find the mean and stdev of the height to width ratio for Gaussian Naive Bayes.
		height_to_width_ratios = pd.DataFrame([calculate_height_and_width(row[1], "height_to_width_ratio") for row in group.iterrows()])
		pose.height_to_width_ratio_means = height_to_width_ratios.mean()
		pose.height_to_width_ratio_stdevs = height_to_width_ratios.std()
	
	if "discretized_height_to_width_ratio" in mode:
		#Find the discretized height to width ratio for Categorical Naive Bayes.
		discretized_height_to_width_ratios = pd.DataFrame([calculate_height_and_width(row[1], "discretized_height_to_width_ratio") for row in group.iterrows()])
		#Using add one laplace smoothing.
		counts = discretized_height_to_width_ratios.value_counts()
		pose.discretized_height_to_width_ratio_probs = np.zeros(3) + 1
		pose.discretized_height_to_width_ratio_probs[list(map(lambda x: int(x[0]), counts.index))] += counts.values
		pose.discretized_height_to_width_ratio_probs /= discretized_height_to_width_ratios.count()[0]
	
	if "closest_points" in mode:
		pose.closest_point_probs = {}
		for n in parameters['closest_points']:
			#Find the closest point of every point in the data frame. 
			#The input is a (len(group)x22) data frame. The output is a (len(group)x11) data frame.
			closest_points = pd.DataFrame([calculate_closest_points(row[1], n) for row in group.iterrows()])
			#Then the probability of each point being the closest point for each body point is calculated for categorical Naive Bayes.
			#Using Laplace add 1 smoothing.
			pose.closest_point_probs[n] = np.zeros((11,12)) + 1
			for column_index in closest_points:
				counts = closest_points[column_index].value_counts()
				pose.closest_point_probs[n][column_index][counts.index] += counts.values
			#Getting rid of the NA value counts.
			pose.closest_point_probs[n] = pose.closest_point_probs[n][:,:-1]
			pose.closest_point_probs[n] = pose.closest_point_probs[n]/(np.sum(pose.closest_point_probs[n], axis = 1)).reshape(11,1)
		
	if "arms_above_head" in mode:
		#Attribute representing if 0, 1 or 2 arms are above the head.
		num_arms_above_head = group.apply(calculate_num_arms_above_head, axis=1)
		counts = num_arms_above_head.value_counts()
		#Using Laplace add 1 smoothing.
		pose.arms_above_head_probs = np.zeros(3) + 1
		pose.arms_above_head_probs[counts.index.astype('int')] += counts.values
		pose.arms_above_head_probs /= sum(pose.arms_above_head_probs)
		
	if "perpendicular_torso" in mode:
		#Calculating the probability that the torso is perpendicular for Categorical Naive Bayes.
		alignment_diffs = list(group.apply(calculate_perpendicular_torso, axis=1))
		#Using Laplace add 1 smoothing.
		prob_straight = (np.nansum(alignment_diffs) + 1)/len(alignment_diffs)
		prob_not_straight = (len(alignment_diffs) - np.nansum(alignment_diffs) + 1)/len(alignment_diffs)
		pose.perpendicular_torso_probs = [prob_not_straight, prob_straight]
		
	if "distance_between_points" in mode:
		#Calculating the distance between all points for Gaussian Naive Bayes.
		distances = pd.DataFrame(calculate_distance_between_points(row[1]) for row in group.iterrows())
		pose.distance_means = distances.mean()
		pose.distance_stdevs = distances.std()
		
	if "key_angles" in mode:
		#Calculating the key angles between points for Gaussian Naive Bayes.
		angles = pd.DataFrame(calculate_key_angles(row[1]) for row in group.iterrows())
		pose.angle_means = angles.mean()
		pose.angle_stdevs = angles.std()
		
	if "discretized_key_angles" in mode:
		#Find the discretized key angles for Categorical Naive Bayes.
		angles = pd.DataFrame(calculate_key_angles(row[1]) for row in group.iterrows())
		angles = (angles/60).apply(np.floor)
		
		pose.discretized_angle_probs = np.zeros((7,6))+1
		for col in angles:
			#Using Laplace add 1 smoothing.
			counts = angles[col].value_counts()
			pose.discretized_angle_probs[col][counts.index.astype('int')] += counts.values
		pose.discretized_angle_probs /= angles.count().to_numpy().reshape(7,1)
		
	return pose

#Training: Determines priors and attribute distributions for every class.
#Returns a pandas series that contains pose objects for every pose.
#Each object contains priors and attribute distributions.
def train(data, mode, parameters):
	poses = [calculate_model_info(data.loc[group[1].index], len(data), mode, parameters) for group in data.groupby([0])]
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
def random_hold_out(data, hold_out_percent, mode, parameters, speedup):
	train_data = data.sample(frac = hold_out_percent, random_state = 3)
	test_data = data.drop(train_data.index)
	poses = train(train_data, mode)
	predictions = predict(test_data, poses, mode, parameters, speedup)
	print(evaluate(predictions, test_data))
	
#Cross validation.
def cross_validation(data, num_partitions, mode, parameters, speedup):
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
		poses = train(train_data, mode, parameters)
		predictions = predict(test_data, poses, mode, parameters, speedup)
		accuracy += evaluate(predictions, test_data)
	return accuracy/num_partitions
	
#Graph displaying the accuracy of KDE at a variety of bandwidths.
#The function returns the bandwidth that maximised accuracy.
def optimize_bandwidth(data, num_partitions, min_bandwidth, max_bandwidth, step, speedup):
	accuracies = []
	bandwidths = np.arange(min_bandwidth, max_bandwidth+step, step)
	for bandwidth in bandwidths:
		accuracy = cross_validation(data, num_partitions, "KDE", [bandwidth], speedup)
		accuracies.append(accuracy)
		print(f"Bandwidth being tested: {bandwidth}")
	
	plt.plot(bandwidths, accuracies, '-o')
	plt.title('Accuracy vs Bandwidth')
	plt.xlabel('Bandwidth value')
	plt.ylabel('% Accuracy')
	colours = ['0.7', '0.5']*(int((max_bandwidth - min_bandwidth + step)/(2*step)))
	if (max_bandwidth - min_bandwidth + 1)%2:
		colours.append('0.7')
	plt.vlines(bandwidths, min(accuracies) - 1, accuracies, linestyles = 'dashed', color = colours)
	plt.show()
	
	return bandwidths[np.argmax(accuracies)]
	
#Function that simplifies modes for cross validation
def select_modes_and_params():
	modes = {0: "NULL", 1: "classic", 2: "KDE", 3: "coordinate_absence", 4: "coordinate_presence", 
	5: "height_and_width", 6: "height_to_width_ratio", 7: "discretized_height_to_width_ratio",
	8: "closest_points", 9: "arms_above_head", 10: "perpendicular_torso", 
	11: "distance_between_points", 12: "key_angles", 13: "discretized_key_angles", 14: "integration"}	
	print(modes, '\n')

	selected_modes = input("Enter the numbers separated by spaces: ")
	selected_modes = list(map(int, selected_modes.split()))
	selected_modes = [modes[mode] for mode in selected_modes]
	
	parameters = {}
	
	KDE_param =  input("Enter parameters for KDE, or leave blank: ")
	closest_points_param = input("Enter parameters for closest points, or leave blank: ")
	epsilon_param = input("Enter parameters for epsilon when integrating, or leave blank: ")
	
	if KDE_param: parameters['KDE'] = int(KDE_param)
	if closest_points_param: parameters['closest_points'] = list(map(int, closest_points_param.split()))
	if epsilon_param: parameters['epsilon'] = float(epsilon_param)
	
	speedup = int(input("Speedup? 0 for No , 1 for Yes: "))
	
	return selected_modes, parameters, speedup
		
def select_modes_and_crossvalidate():
	data = preprocess('train.csv')
	selected_modes, parameters, speedup = select_modes_and_params()	
	print('\n', selected_modes)
	num_partitions = input("Select number of partitions for cross validation: ")
	print(cross_validation(data, 5, selected_modes, parameters, speedup))

def predictions_comparison(ground_truth, predictions1, predictions2):
	predictions1 = pd.Series(predictions1)
	predictions2 = pd.Series(predictions2)
	ground_truth = pd.Series(ground_truth)
	
	predictions_both_got_right = ((ground_truth == predictions1) & (ground_truth == predictions2))
	predictions_both_got_wrong = ((ground_truth != predictions1) & (ground_truth != predictions2))
	predictions_different = (predictions1 != predictions2)
	
	predictions1_different_and_right = predictions1.loc[predictions_different] == ground_truth.loc[predictions_different]
	predictions1_different_and_wrong = predictions1.loc[predictions_different] != ground_truth.loc[predictions_different]
	predictions2_different_and_right = predictions2.loc[predictions_different] == ground_truth.loc[predictions_different]
	predictions2_different_and_wrong = predictions2.loc[predictions_different] != ground_truth.loc[predictions_different]
	
	print(f"Both right: {predictions_both_got_right.sum()}, Both wrong: {predictions_both_got_wrong.sum()}")
	print(f"Different predictions: {predictions_different.sum()}, Total size: {len(ground_truth)}")
	print(f"Of the different predictions, the first set of predictions had {predictions1_different_and_right.sum()} correct and {predictions1_different_and_wrong.sum()} wrong")
	print(f"Of the different predictions, the second set of predictions had {predictions2_different_and_right.sum()} correct and {predictions2_different_and_wrong.sum()} wrong")
	

def compare_predictions_between_modes():
	data = preprocess('train.csv')
	test = preprocess('test.csv')
	
	print("Select the arguments for the first set of predictions")
	modes, parameters, speedup = select_modes_and_params()
	poses = train(data, modes, parameters)
	predictions1 = predict(test, poses, modes, parameters, speedup)
	
	print("\n Select the arguments for the second set of predictions")
	modes, parameters, speedup = select_modes_and_params()
	poses = train(data, modes, parameters)
	predictions2 = predict(test, poses, modes, parameters, speedup)
	
	print()
	predictions_comparison(test[0], predictions1, predictions2)
