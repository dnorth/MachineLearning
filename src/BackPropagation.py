from math import exp
import numpy as np

class BackPropLayer:
	def __init__(self, learningRate, bias):
		self.weights = None
		#self.deltaWeight = np.zeros(weightLen)
		self.learningRate = learningRate
		self.inputs = None
		self.bias = bias
		self.deltaWeights = None
	def get_output_sigmoid(self, data):
		return 1 / ( 1 + exp ( -data ) )
	
	def update_weights(self):
		self.weights = self.weights + self.deltaWeights


class OutputLayer(BackPropLayer):
	def __init__(self, learningRate=1, bias=1):
		BackPropLayer.__init__(self, learningRate, bias)
		self.target = None
		self.output = None

	def update_inputs(self, inputs):
		if not isinstance(self.weights, (np.ndarray, np.generic)):  #Create new weights if this is the first time updating inputs
			self.weights = np.ones(len(inputs) + 1) # + 1 to include the bias weight
			print "\nCREATING DELTAS\n"
			self.deltaWeights = np.zeros(len(inputs) + 1)
		self.inputs = np.array(inputs)

	def update_target(self, target):
		self.target = target

	def get_net(self, data):
		count = 0
		for i in xrange(len(data)):
			count += data[i] * self.weights[i] 

		#Include the bias in prediction
		count += self.bias * self.weights[-1]
		return count

	def calculate_output(self):
		net = self.get_net(self.inputs)
		self.output = self.get_output_sigmoid( net )

	def calculate_error(self):
		return (self.target - self.output) * self.output * ( 1 - self.output )

	def calculate_delta_weights(self, outputError, momentum=0):
		inputs_and_bias = np.append(self.inputs, self.bias)
		self.deltaWeights = self.learningRate * outputError * inputs_and_bias + momentum * self.deltaWeights

class HiddenLayer(BackPropLayer):
	def __init__(self, learningRate=1, bias=1):
		BackPropLayer.__init__(self, learningRate, bias)
		self.outputs = []
		self.inputNets = []

	def update_inputs(self, inputs):
		if not isinstance(self.weights, (np.ndarray, np.generic)):  #Create new weights if this is the first time updating inputs
			self.weights = np.ones([ len(inputs), len(inputs) + 1]) # + 1 to include the bias weight
			self.deltaWeights = np.zeros([ len(inputs), len(inputs) + 1])
		self.inputs = np.array(inputs)

	def get_net (self, input_and_bias):
		input_and_bias = np.atleast_2d( input_and_bias )
		return input_and_bias.dot( self.weights.T )

	def calculate_outputs(self):
		self.outputs = np.array([ self.get_output_sigmoid(x) for x in self.inputNets ])

	def update_input_net(self, inputNets):
		self.inputNets = inputNets

	def calculate_error(self, outputError, outputWeights):
		return self.outputs * (1 - self.outputs) * outputError * outputWeights

	def calculate_delta_weights(self, hiddenErrors, inputs_and_bias, momentum=0):
		for x in xrange(len(self.weights)):
			self.deltaWeights[x] = self.weights[x] * hiddenErrors[x] * inputs_and_bias + momentum * self.deltaWeights[x]


class BackPropNode:
	def __init__(self):
		self.hidden_layer = HiddenLayer() #only accounts for 1 hidden layer
		self.output_layer = OutputLayer()

	def train(self, instances):
		targets = instances[:, -1]
		inputs = instances[:, :-1]

		for x in xrange(len(inputs)):
			self.forwardMotion( inputs[x], targets[x])
			self.backProp()

	def predict(self, inputs):
		return 1

	def forwardMotion(self, feature, target):
		# Assuming just 1 hidden layer
		self.hidden_layer.update_inputs( feature )
		inputs_and_bias = np.append(self.hidden_layer.inputs, self.hidden_layer.bias)
		inputNets = self.hidden_layer.get_net(inputs_and_bias)[0] 
		self.hidden_layer.update_input_net( inputNets )
		self.hidden_layer.calculate_outputs()
		self.output_layer.update_inputs( self.hidden_layer.outputs )
		self.output_layer.update_target( target )
		self.output_layer.calculate_output()

	def backProp(self):
		outputError = self.output_layer.calculate_error()
		self.output_layer.calculate_delta_weights(outputError)
		hiddenErrors = self.hidden_layer.calculate_error(outputError, self.output_layer.weights[:-1])
		inputs_and_bias = np.append(self.hidden_layer.inputs, self.hidden_layer.bias)
		self.hidden_layer.calculate_delta_weights(hiddenErrors, inputs_and_bias)
		self.output_layer.update_weights()
		self.hidden_layer.update_weights()
		print "New Output Weights: \n\n", self.output_layer.weights
		print "New Hidden Weights: \n\n", self.hidden_layer.weights



