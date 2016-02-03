from Perceptron import Perceptron 
from numpy import *
import arff
from collections import OrderedDict

def read_set(filename):
	dataset = arff.load(open(filename))
	data = array(dataset['data'])
	attribute_types = list(OrderedDict(dataset['attributes']).values())
	attribute_names = list(OrderedDict(dataset['attributes']).keys())
	num_classes = len(attribute_types[-1])
	class_names = attribute_types[-1]
	feature_names = attribute_names[:-1]
	feature_class_names = attribute_types[:-1]

	num_classes_per_feature = [len(attribute) for attribute in attribute_types[:-1]]

	numericdata = data

	for row in numericdata:
		for c in range(len(row)):
			if type(attribute_types[c]) is list:
				if row[c] in attribute_types[c]:
					row[c] = attribute_types[c].index(row[c])
				#else:
				#    row[c] = NaN

			elif row[c] is None:
				row[c] = NaN

	return asarray(numericdata, 'float64'), num_classes, num_classes_per_feature, \
					feature_names, feature_class_names, class_names, attribute_types


def iris_test(algorithms, test_set, num_classes, specifiedGoal=100):
	confusion_matrix = asarray(zeros((num_classes, num_classes)))

	num_correct = 0.0
	for instance in test_set:
		features = instance[:-1]
		goal = instance[-1]
		predictions = {}
		for algorithm in algorithms:
			prediction = algorithm.predict(features)
			if prediction == 1:
				predictions[algorithm.specifiedTarget] = algorithm.get_net(features)
		
		prediction = 0
		maxVal = -1
		for key, val in predictions.items():
			if val > maxVal:
				maxVal = val
				prediction = key


		num_correct += 1 if prediction == goal else 0

		confusion_matrix[int(goal), int(prediction)] += 1

	accuracy = (num_correct / len(test_set)) * 100

	return accuracy, confusion_matrix

def main():

	filename = '../TestData/Perceptron/irisFull.arff'

	instances, num_classes, num_classes_per_feature, feature_names, feature_class_names, class_names, attribute_types = read_set(filename)
	training_set = instances
	test_set = instances

	setosaP = Perceptron(threshold=0, learningRate=.1, specifiedTarget=0)
	versicolorP = Perceptron(threshold=0, learningRate=.1, specifiedTarget=1)
	virginicaP = Perceptron(threshold=0, learningRate=.1, specifiedTarget=2)


	setosaP.train(training_set)
	versicolorP.train(training_set)
	virginicaP.train(training_set)

	algorithms = [setosaP, versicolorP, virginicaP]

	accuracy, confusion_matrix = iris_test(algorithms, test_set, num_classes, specifiedGoal=0)
	print("accuracy: %.3f%%" % accuracy)
	print(confusion_matrix)



if __name__ == '__main__':
	main()
