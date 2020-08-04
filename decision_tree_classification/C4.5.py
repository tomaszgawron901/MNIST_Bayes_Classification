from data import MNIST_Data_Loader as Loader

import numpy as np
import scipy.stats


def entropy(labels):
    size = np.size(labels)
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / size
    return scipy.stats.entropy(probabilities, base=10)


def information_lose(labels, bool_array):
    total_size = np.size(labels)
    positive_size = np.sum(bool_array)
    negative_size = total_size - positive_size
    return positive_size / total_size * entropy(labels[bool_array]) + \
           negative_size / total_size * entropy(labels[~bool_array])


def the_best_slicing(data, labels):
    partitions = np.unique(data)

    chosen_bool_array = np.zeros(shape=(), dtype=np.bool)
    min_info_lose = entropy(labels)
    chosen_partition = partitions[0]
    for partition in partitions[1:]:
        bool_array = data < partition
        info_lose = information_lose(labels, bool_array)
        if info_lose < min_info_lose:
            chosen_bool_array = bool_array
            min_info_lose = info_lose
            chosen_partition = partition

    return min_info_lose, chosen_partition, chosen_bool_array


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

        entr = entropy(labels)
        lose_threshold = entr - threshold  # lose has to be less than that.

        chosen_column = 0
        min_lose, chosen_partition, chosen_bool_array = the_best_slicing(data[:, 0], labels)
        loses = np.empty(shape=d, dtype=np.float)
        loses[0] = min_lose
        for column in range(1, d):
            lose, partition, bool_array = the_best_slicing(data[:, column], labels)
            loses[column] = lose
            if lose < min_lose:
                chosen_column = column
                min_lose = lose
                chosen_partition = partition
                chosen_bool_array = bool_array

        if min_lose > lose_threshold:
            unique, count = np.unique(labels, return_counts=True)
            label = unique[np.argmax(count)]
            new_leaf = Leaf(label)
            node_setter(new_leaf)
        else:
            indices_strainer = loses <= lose_threshold
            new_node = Node(index=indices[chosen_column], partition=chosen_partition)
            node_setter(new_node)
            n_chosen_bool_array = ~chosen_bool_array

            self._fit(data[chosen_bool_array][:, indices_strainer], labels[chosen_bool_array], indices[indices_strainer], new_node.set_less, threshold)
            self._fit(data[n_chosen_bool_array][:, indices_strainer], labels[n_chosen_bool_array], indices[indices_strainer], new_node.set_greater, threshold)

    def train(self, X: np.ndarray, y: np.ndarray, threshold=0.):
        n, d = np.shape(X)
        self._fit(X, y, np.arange(d), self._set_root, threshold)

    def classify(self, X: np.ndarray):
        n, d = np.shape(X)
        return np.fromiter((self.root.classify(X[row]) for row in range(n)), np.int)


if __name__ == '__main__':
    import wandb
    import time

    parameters = dict(
        training_data_size=20000,
        testing_data_size=3000,
        threshold=1e-2
    )

    wandb.init(config=parameters, name="C45_Prototype-bigdata")

    tree = C45DecisionTree()

    X, y = Loader.get_training_data('../data')
    print("C45 training started with {} training data.".format(parameters['training_data_size']))
    training_start = time.process_time()
    tree.train(X[:parameters['training_data_size']], y[:parameters['training_data_size']],
               threshold=parameters['threshold'])
    training_time = time.process_time() - training_start
    print("Naive Bayes classification training finished. Time {}s.".format(training_time))

    Xt, yt = Loader.get_test_data('../data')
    print("C45 evaluation started with {} testing data".format(parameters['testing_data_size']))
    evaluation_start = time.process_time()
    score = np.mean(tree.classify(Xt[:parameters['testing_data_size']]) == yt[:parameters['testing_data_size']])
    evaluation_time = time.process_time() - evaluation_start
    print("Evaluation finished with accuracy {:0.3f}%. Time {}s.".format(score * 100, evaluation_time))

    wandb.run.summary['Accuracy'] = score
    wandb.run.summary['Training Time'] = training_time
    wandb.run.summary['Evaluation Time'] = evaluation_time
