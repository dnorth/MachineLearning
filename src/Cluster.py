import random
import numpy as np

class Cluster:
	def __init__(self, k, maxIterations=15, normalize=False):
		self.k = k
		self.training_instances = None
		self.training_targets = None
		self.centroids = []
		self.nData = None
		self.nDim = None
		self.normalize = normalize
		self.maxIterations = maxIterations

	def train(self, instances):
		self.training_targets = instances[:, -1]
		self.training_instances = instances[:, :-1] #Get everything except the 0 index, which is the row index

		
		self.nData, self.nDim = np.shape(self.training_instances)

		self.training_instances = [[col if not np.isnan(col) else 14 for col in row] for row in self.training_instances]



		if self.normalize:
			self.training_instances = self.normalize_data(self.training_instances)

		for index in xrange(self.k):
			self.centroids.append( self.training_instances[np.random.choice(self.k)] )
		self.centroids = np.array( self.centroids )

		oldCentroids = 40

		count = 0

		while np.sum(np.sum(oldCentroids - self.centroids)) != 0 and count < self.maxIterations:
			oldCentroids = self.centroids.copy()
			distances = self.get_distances(self.training_instances, self.nData)
			cluster = self.get_closest_cluster(distances, self.nData)

			for j in range(self.k):
				thisCluster = np.where(cluster==j,1,0)
				if sum(thisCluster)>0:
					self.centroids[j,:] = np.sum(self.training_instances*thisCluster,axis=0)/np.sum(thisCluster)
			
			sse = 0
			for x in xrange(len(cluster)):
				if np.isnan(self.training_targets[x]):
					self.training_targets[x] = 1
				diff = cluster[x] - self.training_targets[x]
				sse += (diff) ** 2

			#print("Iter: %d SSE: %.3f" % (count, sse))

			count += 1

		#print "Final Centroids: \n\n"
		#self.print_centroids()


	def normalize_data(self, data):
		data = data[:] - np.nanmin(data, axis=0)
		return data[:] / np.nanmax(data, axis=0)		


	def predict(self, inputs):
		#inputs = inputs[1:]
		inputs = [col if not np.isnan(col) else 1 for col in inputs]
		nData = np.shape(inputs)[0]

		distances = []

		for j in range(self.k-1):
			distances.append( np.sum((np.array(inputs) - np.array(self.centroids)[j]) **2 , axis=0))
		
		return np.argmin(distances, axis=0)

	def print_centroids(self):
		for index in xrange(self.k):
			print "Centroid ", index, " = ", self.centroids[index]

	def get_distances(self, data, nData):
		distances = np.ones((1,nData))*np.sum((np.array(data)-np.array(self.centroids)[0,:])**2,axis=1)


		for j in range(self.k-1):
				distances = np.append(distances,np.ones((1,nData))*np.sum((np.array(data)-np.array(self.centroids)[j+1,:])**2,axis=1),axis=0)
		
		return distances

	def get_closest_cluster(self, distances, nData):
		cluster = distances.argmin(axis=0)
		cluster = np.transpose(cluster*np.ones((1,nData)))

		return cluster
