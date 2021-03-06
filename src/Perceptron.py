import numpy as np
import random 

#Perceptron Node
class Perceptron:
	def __init__(self, threshold=0, learningRate = 0.1, bias=1, specifiedTarget=100):
		self.threshold = threshold
		self.weights = None
		self.deltaWeight = None
		self.bias = bias
		self.specifiedTarget = specifiedTarget
		self.learningRate = learningRate
		#self.epochDeltas = []   #Used for calculating change over epochs
	#Takes in an array of inputs and a target for those inputs
	#Changes the current weights
	def trainSet(self, inputs, target):
		#print "Inputs: ", inputs
		output = self.predict(inputs)
		#print "Output: ", output, " Target: ", target, " Equal: ", output == target
		if output != target:
			for i in xrange(len(inputs)):
				self.deltaWeight[i] =  self.learn(inputs[i], target, output)
				self.weights[i] += self.deltaWeight[i]
			#Recalculate the bias as well
			self.deltaWeight[-1] = self.learn(self.bias, target, output)
			self.weights[-1] += self.deltaWeight[-1]
		#print "Delta W: ", self.deltaWeight

	def checkThreshold(self, final):
		return 1 if final > self.threshold else 0

	def learn(self, input, target, output):
		return self.learningRate * (target - output) * input

	def train(self, instances):
		weightLen = len(instances[0]) #Full length to include the bias
		self.weights = np.zeros(weightLen) #np.random.rand(inputSize + 1) 
		self.deltaWeight = np.zeros(weightLen)

		targets = instances[:, -1]

		if self.specifiedTarget != 100:  #If we specify a target, we're using more than one perceptron
			targets = [1 if x == self.specifiedTarget else 0 for x in targets]

		inputs = instances[:, :-1]

		#how many epochs?
		count = 0
		zeroChange = 0
		epochWeight = np.ones(weightLen)
		while zeroChange <= 3  and count <= 1000:  #don't stop until we have 0 change 3 consecutive times or we reach 1000 epochs
			epochWeight =  np.zeros(weightLen)

			#shuffle the data after every epoch
			combined = zip(inputs, targets)
			np.random.shuffle(combined)
			inputs = []
			targets = []
			inputs[:], targets[:] = zip(*combined)

			for x in xrange(len(targets)):
				self.deltaWeight = np.zeros(weightLen)
				self.trainSet(inputs[x], targets[x])
				epochWeight += self.deltaWeight
			#print "Epoch Weight: " , epochWeight
			#print "Epoch Delta: " , pow(np.mean(epochWeight) , 2 ) 
			
			#Used to plot the change over epochs
			#self.epochDeltas.append( pow(np.mean(epochWeight) , 2 ) )

			if pow(np.mean(epochWeight) , 2 ) == 0:
				zeroChange += 1
			else:
				zeroChange = 0
			count += 1
		print "Global Change: " , self.weights
		print "Trained after ", count - 1, " Epochs."

	def predict(self, inputs):
		count = self.get_net(inputs)
		return self.checkThreshold(count)
		
	def get_net(self, inputs):
		count = 0
		for i in xrange(len(inputs)):
			count += inputs[i] * self.weights[i] 

		#Include the bias in prediction
		count += self.bias * self.weights[-1]
		return count