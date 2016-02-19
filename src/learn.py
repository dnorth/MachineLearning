from numpy import *
from enum import Enum
from collections import OrderedDict
import arff
# from MultilayerPerceptron import MultilayerPerceptron
# from DecisionTree import DecisionTree
from Perceptron import Perceptron
from BackPropagation import BackPropNode
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
    if algorithm_name == 'backprop':
        algorithm = BackPropNode(hidden_node_multiplier=6, rand_weights=True, learning_rate=0.3, momentum=0.2, num_outputs=1)
    if algorithm_name == 'knn':
        algorithm = KNearestNeighbors(k=15, distance_weighting=False, normalize=False)
    elif algorithm_name == 'knn_regression':
        algorithm = KNearestNeighbors(k=3, regression=True, distance_weighting=True)
    elif algorithm_name == 'knn_mixed':
        algorithm = KNearestNeighbors(k=3, distance_weighting=True, attribute_types=attribute_types)

    if algorithm_name =='knn':
        accuracies = []
        k_values = arange(1, 16, 2)
        algorithm.train(training_set)

        for k in k_values:
            algorithm.k = k
            accuracy, confusion_matrix = test(algorithm, test_set, num_classes)
            print("k: %d, accuracy: %.3f%%" % (k, accuracy))
            accuracies.append(accuracy)

        figure()
        plot(k_values, accuracies)
        xticks(k_values)
        title('Magic Telescope Test Set Accuracy')
        xlabel('k')
        ylabel('Accuracy')
        tight_layout()

        show()

    elif algorithm_name == 'knn_regression':
        mses = []
        k_values = arange(1, 16, 2)

        algorithm.train(training_set)

        for k in k_values:
            algorithm.k = k
            mse = test_continuous(algorithm, test_set)
            print("k: %d, MSE: %.3f" % (k, mse))
            mses.append(mse)

        figure()
        plot(k_values, mses)
        xticks(k_values)
        title('Housing Price Prediction Test Error')
        xlabel('k')
        ylabel('MSE')
        tight_layout()

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

def cross_validate(algorithm, data, num_classes, num_folds=10, return_training_accuracy=True):
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

    if return_training_accuracy:
        return training_accuracies, test_accuracies
    else:
        return test_accuracies


def test(algorithm, test_set, num_classes):
    confusion_matrix = asarray(zeros((num_classes, num_classes)))

    num_correct = 0.0
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


def test_continuous(algorithm, test_set):
    num_correct = 0.0
    sse = 0.0
    for instance in test_set:
        features = instance[:-1]
        goal = instance[-1]

        prediction = algorithm.predict(features)

        sse += (goal - prediction) ** 2

    mse = sse / len(test_set)

    return mse

if __name__ == '__main__':
    main()
