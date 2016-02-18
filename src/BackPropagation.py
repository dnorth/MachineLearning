from math import exp
import matplotlib.pyplot as plt 
import numpy as np
import time
import random

class BackPropLayer:
	def __init__(self, learning_rate, bias, rand_weights, momentum):
		self.weights = None
		self.learning_rate = learning_rate
		self.inputs = None
		self.bias = bias
		self.deltaWeights = None
		self.rand_weights = rand_weights
		self.momentum = momentum
	def get_output_sigmoid(self, data):
		return 1 / ( 1 + exp ( -data ) )
	
	def update_weights(self):
		self.weights = self.weights + self.deltaWeights

	def get_net(self, input_and_bias):
		input_and_bias = np.atleast_2d( input_and_bias )
		return input_and_bias.dot( self.weights.T )

	def calculate_outputs(self, nets):
		self.outputs = np.array([ self.get_output_sigmoid(x) for x in nets ])

	def calculate_delta_weights(self, errors):
		inputs_and_bias = np.append(self.inputs, self.bias)
		for x in xrange(len(self.weights)):
			#print "DeltaWeights: ", self.deltaWeights[x], " Learning_Rate: ", self.learning_rate
			self.deltaWeights[x] = self.learning_rate  * errors[x] * inputs_and_bias + self.momentum * self.deltaWeights[x]


class OutputLayer(BackPropLayer):
	def __init__(self, rand_weights, learning_rate, momentum, bias, num_outputs):
		BackPropLayer.__init__(self, learning_rate, bias, rand_weights, momentum)
		self.target = None
		self.outputs = None
		self.num_outputs = num_outputs

	def update_inputs(self, inputs):
		if not isinstance(self.weights, (np.ndarray, np.generic)):  #Create new weights if this is the first time updating inputs
			rowLen = 1 * self.num_outputs
			colLen = len(inputs) + 1 # + 1 to include the bias weight
			if self.rand_weights:
				self.weights = np.random.uniform(-1, 1, [rowLen, colLen])
			else:
				#self.weights = np.array([ -0.01, 0.03, 0.02, 0.02 ])
				self.weights = np.ones([rowLen, colLen]) # + 1 to include the bias weight
			self.deltaWeights = np.zeros([rowLen, colLen])
		self.inputs = np.array(inputs)

	def update_target(self, target):
		self.target = target

	def calculate_error(self):
		return (self.target - self.outputs) * self.outputs * ( 1 - self.outputs )

class HiddenLayer(BackPropLayer):
	def __init__(self, hidden_node_multiplier, rand_weights, learning_rate, momentum, bias):
		BackPropLayer.__init__(self, learning_rate, bias, rand_weights, momentum)
		self.outputs = []
		self.hidden_node_multiplier = hidden_node_multiplier

	def update_inputs(self, inputs):
		if not isinstance(self.weights, (np.ndarray, np.generic)):  #Create new weights if this is the first time updating inputs
			rowLen = len(inputs) * self.hidden_node_multiplier
			colLen = len(inputs) + 1 # + 1 to include the bias weight
			if self.rand_weights:
				self.weights = np.random.uniform(-1, 1, [ rowLen , colLen ])
			else:
				#self.weights = np.array([ [-0.03, 0.03, -0.01], [0.04, -0.02, 0.01], [0.03, 0.02, -0.02] ])
				self.weights = np.ones([ rowLen , colLen ]) 
			self.deltaWeights = np.zeros([ rowLen , colLen ])
		self.inputs = np.array(inputs)

	def calculate_error(self, outputErrors, outputWeights):
		errors = np.zeros( [ len(outputWeights), len(self.outputs)])
		for x in xrange(len(outputErrors)):
			errors[x] = self.outputs * (1 - self.outputs) * outputErrors[x] * outputWeights[x][:-1]
		return errors[0]

class BackPropNode:
	def __init__(self, hidden_node_multiplier=1, rand_weights=False, learning_rate=1, momentum=0, bias=1, num_outputs=1):
		self.hidden_layer = HiddenLayer(hidden_node_multiplier, rand_weights, learning_rate, momentum, bias) #only accounts for 1 hidden layer
		self.output_layer = OutputLayer(rand_weights, learning_rate, momentum, bias, num_outputs)

	def train(self, instances):
		percent_for_training = .80
		
		training_instances = instances[0:percent_for_training * len(instances)]
		validation_instances = instances[percent_for_training * len(instances):-1]

		targets = training_instances[:, -1]
		inputs = training_instances[:, :-1]

		bssf = 0
		bssfStable = 0
		epochNum = 0

		while( bssfStable < 1000 and (bssf < 100 or bssfStable < 5) ):
		 	epochNum += 1
		 	#shuffle the data after every epoch
		 	inputs, targets = self.shuffle_data(inputs, targets)
		 	outputs = []
			for x in xrange(len(inputs)):
				self.output_layer.update_target( targets[x] )
				output = self.forwardMotion( inputs[x] )
				outputs.append(output)
				self.backProp()
		 	accuracy = self.validate_data(validation_instances)
		 	if accuracy > bssf:
		 		bssf = accuracy
		 		bssfStable = 0
		 	else:
		 		bssfStable += 1
		print "Epoch: ", epochNum
			# Plotting for the Y values
			#if e in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]:
			#	plt.title("Epoch %d" % e)
			#	plt.scatter( inputs[:,0], inputs[:,1], color=map(str, outputs), s = 1500, marker='s')#, edgecolor='black')
			#	plt.gray()
			#	plt.clim(0,1)
			#	plt.show()

	def validate_data(self, validation_set):
		num_correct = 0.0
		for instance in validation_set:
			features = instance[:-1]
			goal = instance[-1]

			prediction = self.predict(features)
			num_correct += 1 if prediction == goal else 0

		accuracy = (num_correct / len(validation_set)) * 100
		return accuracy

	def shuffle_data(self, inputs, targets):
		combined = zip(inputs, targets)
		np.random.shuffle(combined)
		inputs = []
		targets = []
		inputs[:], targets[:] = zip(*combined)
		return inputs, targets

	def predict(self, inputs):
		prediction = self.forwardMotion( inputs )
		if len(prediction) == 1:
			if prediction < 0.4:
				return 0
			elif prediction < 0.9994:
				return 1
			else:
				return 2
		else:
			return np.argmax(prediction)

	def forwardMotion(self, feature):
		# Assuming just 1 hidden layer
		self.hidden_layer.update_inputs( feature )
		inputs_and_bias = np.append(self.hidden_layer.inputs, self.hidden_layer.bias)
		input_nets = self.hidden_layer.get_net(inputs_and_bias)[0] 
		self.hidden_layer.calculate_outputs(input_nets)
		#print "Sigmoid Output: ", self.hidden_layer.outputs
		self.output_layer.update_inputs( self.hidden_layer.outputs )
		inputs_and_bias = np.append(self.output_layer.inputs, self.output_layer.bias)
		output_nets = self.output_layer.get_net(inputs_and_bias)[0] 
		self.output_layer.calculate_outputs(output_nets)
		return self.output_layer.outputs

	def backProp(self):
		outputError = self.output_layer.calculate_error()
		#print "Output Error: ", outputError
		self.output_layer.calculate_delta_weights(outputError)
		hiddenErrors = self.hidden_layer.calculate_error(outputError, self.output_layer.weights)

		#print "Hidden Errors: ", hiddenErrors
		self.hidden_layer.calculate_delta_weights(hiddenErrors)
		self.output_layer.update_weights()
		self.hidden_layer.update_weights()
		#print "New Output Weights: \n\n", self.output_layer.weights
		#print "New Hidden Weights: \n\n", self.hidden_layer.weights



