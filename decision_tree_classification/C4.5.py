from data import MNIST_Data_Loader as Loader

import numpy as np
import scipy.stats


def entropy(labels):
    size = np.size(labels)
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / size
    return scipy.stats.entropy(probabilities, base=3)


def continuous_gain(data, labels, partition):
    total_size = np.size(labels)
    gain = entropy(labels)

    bool_array = data < partition
    positive_size = np.sum(bool_array)
    negative_size = total_size - positive_size
    gain -= positive_size/total_size * entropy(labels[bool_array])
    gain -= negative_size/total_size * entropy(labels[~bool_array])
    return gain


def maximum_continuous_gain(data, labels, return_partition=False):
    unique_data = np.unique(data)
    gains = np.fromiter((continuous_gain(data, labels, partition) for partition in unique_data), np.float)
    max_gain_index = np.argmax(gains)
    if return_partition:
        return gains[max_gain_index], unique_data[max_gain_index]
    return gains[max_gain_index]


class Node:
    def __init__(self, index, partition):
        self.index = index
        self.partition = partition
        self.greater = None
        self.less = None

    def set_greater(self, node):
        self.greater = node

    def set_less(self, node):
        self.less = node

    def classify(self, x):
        if x[self.index] < self.partition:
            return self.less.classify(x)
        else:
            return self.greater.classify(x)


class Leaf:
    def __init__(self, label):
        self.label = label

    def classify(self, x):
        return self.label


class C45DecisionTree:
    def __init__(self):
        self.root = None

    def _set_root(self, node):
        self.root = node

    def _fit(self, data, labels, indices, node_setter, threshold=0.):
        n, d = np.shape(data)
        info_and_partitions = np.array(
            list(maximum_continuous_gain(data[:, col], labels, return_partition=True) for col in range(d)))
        max_info_index = np.argmax(info_and_partitions[:, 0])
        max_info = info_and_partitions[max_info_index, 0]
        max_info_partition = info_and_partitions[max_info_index, 1]

        if max_info <= threshold:
            unique, count = np.unique(labels, return_counts=True)
            label = unique[np.argmax(count)]
            new_leaf = Leaf(label)
            node_setter(new_leaf)
            print(label)
        else:
            new_node = Node(index=indices[max_info_index], partition=max_info_partition)
            node_setter(new_node)
            bool_array = data[:, max_info_index] < max_info_partition
            n_bool_array = ~bool_array

            self._fit(data[bool_array], labels[bool_array], indices, new_node.set_less, threshold)
            self._fit(data[n_bool_array], labels[n_bool_array], indices, new_node.set_greater, threshold)

    def train(self, X: np.ndarray, y: np.ndarray, threshold=0.):
        n, d = np.shape(X)
        self._fit(X, y, np.arange(d), self._set_root, threshold)

    def classify(self, X: np.ndarray):
        n, d = np.shape(X)
        return np.fromiter((self.root.classify(X[row]) for row in range(n)), np.int)


if __name__ == '__main__':
    training_data_size = 1000
    testing_data_size = 100
    threshold = 1e-2
    tree = C45DecisionTree()

    X, y = Loader.get_training_data('../data')
    print("C45 training started with {} training data.".format(training_data_size))
    tree.train(X[:training_data_size], y[:training_data_size], threshold=threshold)
    print("Naive Bayes classification training finished.")

    Xt, yt = Loader.get_test_data('../data')
    print("C45 evaluation started with {} testing data".format(testing_data_size))
    score = np.mean(tree.classify(Xt[:testing_data_size]) == yt[:testing_data_size])
    print("Evaluation finished with accuracy {:0.3f}%.".format(score*100))


