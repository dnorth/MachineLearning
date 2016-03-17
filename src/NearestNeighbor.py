from __future__ import division
from collections import Counter
import numpy as np

class NearestNeighbor:
	def __init__(self, k, distance_weighting=False, normalize=False, regression=False, attribute_types=None):
		self.k = k
		self.distance_weighting = distance_weighting
		self.normalize = normalize
		self.regression = regression
		self.training_inputs = None
		self.training_targets = None
		self.attribute_types = attribute_types

	def get_euclid_dist(self, test_instance, training_instances):
		return np.sqrt(np.sum((np.array(test_instance) - np.array(training_instances))**2, axis=1)) #Vectorized

	def get_weighted_mean(self, distance_vector):
		return 1 / (distance_vector ** 2)

	def do_weighted_regression(self, weighted_mean_vector, target_vector):
		return np.sum(weighted_mean_vector * target_vector) / np.sum(weighted_mean_vector)

	def get_neighbors(self, test_instance):
		distances = self.get_euclid_dist(test_instance, self.training_instances)
		final_weights = distances
		
		k_smallest_indeces = np.argpartition(distances, self.k)[:self.k]
		
		neighbors = {'targets': [], 'indeces': [], 'distances': []}
		for index in k_smallest_indeces:
			neighbors['indeces'].append(index) 
			neighbors['distances'].append(distances[index])
			neighbors['targets'].append(self.training_targets[index])

		if self.regression:
			if self.distance_weighting:
				weighted_mean_vector = self.get_weighted_mean(np.array(neighbors['distances']))
				final_weight = self.do_weighted_regression(weighted_mean_vector, np.array(neighbors['targets']))
			else:
				final_weight = np.sum(np.array(neighbors['targets'])) / self.k
			return final_weight

		return neighbors

	def most_common(self, lst):
		data = Counter(lst)
		return data.most_common(1)[0][0]

	def train(self, instances):
		self.training_targets = instances[:, -1]
		self.training_instances = instances[:, :-1]

		for index in xrange(len(self.training_instances)):
			self.training_instances[index] = [1 if np.isnan(x) else x for x in self.training_instances[index]]

		if self.normalize:
			self.training_instances = self.normalize_data(self.training_instances)

		#####
		# Commented out test for the experiment portion of the lab
		#####
		#print "Origin Len: ", len(self.training_instances)
		#
		#count = 0
		#while count < len(self.training_instances):
		#	neighbors = self.get_neighbors(self.training_instances[count])
		#	if len(set(neighbors['targets'])) == 1:
		#		self.training_instances = np.delete(self.training_instances, count, 0)
		#		count -= 1
		#	count+= 1
		#
		#print "Modified Len: ", len(self.training_instances)

		#np.random.shuffle(self.training_instances)
		#self.training_instances = self.training_instances[:2000]


	def predict(self, inputs):
		neighbors =  self.get_neighbors(inputs)

		if self.regression:
			return neighbors

		if len(neighbors) == 1:
			return neighbors['targets'][0]
		else:
			if self.distance_weighting:
				means = self.get_weighted_mean(np.array(neighbors['distances']))
				normalizer = np.sum(means)

				votes = {}
				for x in xrange(self.k):
					classifier = neighbors['targets'][x]
					if classifier not in votes:
						votes[classifier] = 0
					votes[classifier] += means[x]

				return np.argmax(np.array(votes.values() / normalizer))

			else:
				return self.most_common(neighbors['targets'])

	def normalize_data(self, data):
		data = data[:] - np.nanmin(data, axis=0)
		return data[:] / np.nanmax(data, axis=0)
