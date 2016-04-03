from __future__ import division
from numpy import *
from enum import Enum
from collections import OrderedDict
import arff
from NearestNeighbor import NearestNeighbor
from Perceptron import Perceptron
from BackPropagation import BackPropNode
from DecisionTree import DecisionTree
from Cluster import Cluster
from matplotlib.pyplot import *
import matplotlib.patches as mpatches 
import sys


class DatasetType(Enum):
    training, static, random, cross = range(4)

    @classmethod
    def fromstring(cls, str):
        return getattr(cls, str, None)


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


def main():
    filename = '../TestData/Perceptron/linearlySeparable.arff'
    dataset_type = DatasetType.training
    if len(sys.argv) > 1:
        algorithm_name = sys.argv[1]

    if len(sys.argv) > 2:
        filename = sys.argv[2]

    if len(sys.argv) > 3:
        # dataset_type = DatasetType[sys.argv[3]]
        dataset_type = DatasetType.fromstring(sys.argv[3])
        print(dataset_type)

    if dataset_type is DatasetType.training:
        instances, num_classes, num_classes_per_feature, feature_names, feature_class_names, class_names, attribute_types = read_set(filename)
        training_set = instances
        test_set = instances

    elif dataset_type is DatasetType.random:
        instances, num_classes, num_classes_per_feature, feature_names, feature_class_names, class_names, attribute_types = read_set(filename)
        percent_for_training = float(sys.argv[4])
        random.shuffle(instances)
        training_set = instances[0:percent_for_training * len(instances)]
        test_set = instances[percent_for_training * len(instances):-1]

    elif dataset_type is DatasetType.static:
        training_set, num_classes, num_classes_per_feature, feature_names, feature_class_names, class_names, attribute_types = read_set(filename)
        test_set, num_classes, num_classes_per_feature, feature_names, feature_class_names, class_names, attribute_types = read_set(sys.argv[4])

    elif dataset_type is DatasetType.cross:
        data, num_classes, num_classes_per_feature, feature_names, feature_class_names, class_names, attribute_types = read_set(filename)

        random.shuffle(data)

    """
    if algorithm_name == 'multilayer':
        num_features = instances.shape[1] - 1
        algorithm = MultilayerPerceptron((num_features, 16, num_classes), 'classification')
    elif algorithm_name == 'decision':
        algorithm = DecisionTree(num_classes, num_classes_per_feature, feature_names,
                                 feature_class_names, class_names)
    """
    if algorithm_name == 'perceptron':
        algorithm = Perceptron(threshold=0, learningRate=.1)
    elif algorithm_name == 'backprop':
        algorithm = BackPropNode(hidden_node_multiplier=6, rand_weights=True, learning_rate=0.3, momentum=0.2, num_outputs=1)
    elif algorithm_name == 'decisiontree':
        algorithm = DecisionTree(debug=False, validation=False)
    elif algorithm_name == 'knn':
        algorithm = NearestNeighbor(k=7, distance_weighting=False, normalize=True)
    elif algorithm_name == 'knn_regression':
        algorithm = NearestNeighbor(k=3, regression=True, distance_weighting=True, normalize=True)
    elif algorithm_name == 'knn_mixed':
        algorithm = NearestNeighbor(k=13, distance_weighting=False, attribute_types=attribute_types)
    elif algorithm_name == 'cluster' or algorithm_name == 'cluster_mult':
        algorithm = Cluster(k=2, normalize=False)

    if algorithm_name =='knn':
        accuracies = []
        k_values = arange(1,16, 2)
        algorithm.train(training_set)

        import numpy as np

        for k in k_values:
            algorithm.k = k
            accuracy, confusion_matrix = test(algorithm, test_set, num_classes, normalize=True)
            print("k: %d, accuracy: %.3f%%" % (k, accuracy))
            accuracies.append(accuracy)

        figure()
        print "Accuracies: ", accuracies
        plot(k_values, accuracies)
        xticks(k_values)
        title('Magic Telescope Test Set Accuracy')
        xlabel('k')
        ylabel('Accuracy')

        show()

    elif algorithm_name == 'knn_regression':
        mses = []
        k_values = arange(1, 16, 2)

        algorithm.train(training_set)

        for k in k_values:
            algorithm.k = k
            mse = test_continuous(algorithm, test_set, normalize=True)
            print("k: %d, MSE: %.3f" % (k, mse))
            mses.append(mse)

        figure()
        plot(k_values, mses)
        xticks(k_values)
        title('Housing Price Prediction Test Error')
        xlabel('k')
        ylabel('MSE')

        show()

    elif algorithm_name == 'cluster':
        sses = []

        for x in xrange(5):
            algorithm.k = 4
            algorithm.centroids = []
            algorithm.train(training_set)
            sse = test_continuous(algorithm, test_set, normalize=False, only_sse=True)
            sses.append( sse )
            print("k: %d, SSE: %.3f" % (algorithm.k, sse))

        figure()
        plot(xrange(5), sses)
        xticks(xrange(5))
        title('Iris SSE Test Error - K = 4')
        xlabel('Iteration')
        ylabel('SSE')

        show()

    elif algorithm_name == 'cluster_mult':

        k_values = arange(2,8,1)
        sses = []

        for k in k_values:
            algorithm.k = k
            algorithm.centroids = []
            algorithm.train(training_set)
            sse = test_continuous(algorithm, test_set, normalize=False, only_sse=True)
            print("k: %d, SSE: %.3f" % (algorithm.k, sse))
            sses.append(sse)
        
        figure()
        plot(k_values, sses)
        xticks(k_values)
        title('Iris SSE Test Error')
        xlabel('k')
        ylabel('SSE')

        show()


    elif algorithm_name == 'knn_mixed':
        accuracies = []
        # k_values = arange(1, 16, 2)
        test_accuracies = cross_validate(algorithm, data, num_classes, num_folds=10, return_training_accuracy=False)

        print("Test: %s" % test_accuracies)
        print("Test Average Accuracy: %.3f%%" % average(test_accuracies))

    elif dataset_type != DatasetType.cross:
        algorithm.train(training_set)
        targets = instances[:, -1]
        accuracy, confusion_matrix = test(algorithm, test_set, num_classes)
        print("accuracy: %.3f%%" % accuracy)
        print(confusion_matrix)

    else:
        training_accuracies, test_accuracies = cross_validate(algorithm, data, num_classes, num_folds=10)

        print("Training: %s" % training_accuracies)
        print("Training Average Accuracy: %.3f%%" % average(training_accuracies))

        print("Test: %s" % test_accuracies)
        print("Test Average Accuracy: %.3f%%" % average(test_accuracies))

    if algorithm_name == "perceptron":
        figure()

        classDict = OrderedDict()
        for row in test_set:
            if not classDict.has_key(row[-1]):
                classDict[row[-1]] = { "x" : [], "y" : [],  "color" : np.random.rand(3,1), "marker" : np.random.random_integers(3,7)}
            classDict[row[-1]]["x"].append(row[0])
            classDict[row[-1]]["y"].append(row[1])

        #plot_scatter(classDict, algorithm)
        #plot_epoch_change(algorithm)


def plot_scatter(classDict, algorithm):
    legend_info = []
    for class_name, class_info in classDict.items():
        scatter(class_info["x"], class_info["y"], c= class_info["color"], marker= class_info["marker"])
        legend_info.append(mpatches.Patch(color = class_info["color"], label= class_name))

    x, y = get_points_from_weights(algorithm.weights)
    plot(x , y)
    legend(handles = legend_info)
    xlabel('X')
    ylabel('Y')
    show()

def plot_epoch_change(algorithm):
    y_list = algorithm.epochDeltas
    x_list = xrange(len(y_list))

    plot(x_list , y_list)
    xlabel('Epochs')
    ylabel('Change Over Epoch')
    show()

def get_points_from_weights(weights):
    # y = x / yCoeff + b / yCoeff
    x = np.arange(-1, 1, 0.1)
    y = []
    for points in x:
        y.append((-weights[0] * points) / weights[1] + weights[2] / weights[1])
    return x , y

def cross_validate(algorithm, data, num_classes, num_folds=10, return_training_accuracy=True, plot_data=True):
    test_accuracies = []
    training_accuracies = []
    instances_per_fold = int(len(data) / num_folds)
    for fold in range(num_folds):
        mask = zeros(len(data), dtype='bool')
        fold_start = fold * instances_per_fold
        mask[fold_start: fold_start + instances_per_fold] = True
        test_set = data[mask]
        training_set = data[logical_not(mask)]

        algorithm.train(training_set)
        if return_training_accuracy:
            training_accuracy, training_confusion_matrix = test(algorithm, training_set, num_classes)
            training_accuracies.append(training_accuracy)

        test_accuracy, test_confusion_matrix = test(algorithm, test_set, num_classes)
        test_accuracies.append(test_accuracy)

    if plot_data:
        folds_list = xrange(num_folds)

        c1, = plot(folds_list , test_accuracies, color="red", label="test accuracy")
        if return_training_accuracy:
            c2, = plot(folds_list, training_accuracies, color="blue", label="training accuracy")
        xlabel('Fold #')
        ylabel('Accuracy')
        ylim(0,105)

        if return_training_accuracy:
            legend(handles=[c1, c2], loc=0)
        else:
            legend(handles=[c1], loc=0)
        show()

    if return_training_accuracy:
        return training_accuracies, test_accuracies
    else:
        return test_accuracies


def test(algorithm, test_set, num_classes, normalize=False):
    confusion_matrix = asarray(zeros((num_classes, num_classes)))

    num_correct = 0.0

    if normalize:
            test_set = normalize_data(test_set)



    for instance in test_set:
        features = instance[:-1]
        goal = instance[-1]

        prediction = algorithm.predict(features)
        #print "Goal: ", goal
        #print "Prediction: ", prediction
        num_correct += 1 if prediction == goal else 0
        confusion_matrix[int(goal), int(prediction)] += 1

    accuracy = (num_correct / len(test_set)) * 100

    return accuracy, confusion_matrix


def test_continuous(algorithm, test_set, normalize=False, only_sse = False):
    num_correct = 0.0
    sse = 0.0

    for instance in test_set:
        features = instance[:-1]

        if normalize:
            features = normalize_data(features)

        goal = instance[-1]

        prediction = algorithm.predict(features)

        diff = goal - prediction

        if np.isnan(goal) or np.isnan(prediction):
            diff = 1.0

        sse += (diff) ** 2

        #print("Goal: %.2f Prediction: %.2f" % (goal, prediction))

    if only_sse:
        return sse

    mse = sse / len(test_set)

    return mse

def normalize_data(data):
    data = data[:] - np.min(data, axis=0)
    return data[:] / np.max(data, axis=0)

if __name__ == '__main__':
    main()
