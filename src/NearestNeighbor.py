from __future__ import division
from collections import Counter
import numpy as np

class NearestNeighbor:
	def __init__(self, k, distance_weighting=False, normalize=False, regression=False):
		self.k = k
		self.distance_weighting = distance_weighting
		self.normalize = normalize
		self.regression = regression
		self.training_inputs = None
		self.training_targets = None

	def get_euclid_dist(self, test_instance, training_instances):
		return np.sqrt(np.sum((np.array(test_instance) - np.array(training_instances))**2, axis=1)) #Vectorized

	def get_neighbors(self, test_instance):
		distances = self.get_euclid_dist(test_instance, self.training_instances)
		
		k_smallest_indeces = np.argpartition(distances, self.k)[:self.k]
		
		neighbors = []
		for index in k_smallest_indeces:
			neighbors.append( self.training_targets[index] ) #get the next item in distances which should be the smallest
		return neighbors

	def most_common(self, lst):
		data = Counter(lst)
		return data.most_common(1)[0][0]

	def train(self, instances):
		self.training_targets = instances[:, -1]
		self.training_instances = instances[:, :-1]

		if self.normalize:
			self.training_instances = self.normalize_data(self.training_instances)
		
		#np.random.shuffle(self.training_instances)
		#self.training_instances = self.training_instances[:2000]


	def predict(self, inputs):
		neighbors =  self.get_neighbors(inputs)

		if len(set(neighbors)) == 1:
			return neighbors[0]
		else:
			return self.most_common(neighbors)

	def normalize_data(self, data):
		data = data[:] - np.min(data, axis=0)
		return data[:] / np.max(data, axis=0)
