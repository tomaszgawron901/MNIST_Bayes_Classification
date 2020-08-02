from data import MNIST_Data_Loader as Loader

import numpy as np
from scipy import stats


class NonNaiveBayesClassification:
    def __init__(self):
        self.gaussian = []
        self.labels = np.empty(0)

    def train(self, X: np.ndarray, y: np.ndarray, smoothing=1.):
        self.labels = np.unique(y)
        n, d = X.shape
        for l in self.labels:
            divided_data = X[y == l]
            x_mean = np.mean(divided_data, axis=0)
            cov = np.cov(divided_data.T) + np.eye(d)*smoothing
            self.gaussian.append({'x_mean': x_mean, 'cov': cov})

    def classify(self, X: np.ndarray):
        n, d = np.shape(X)
        k = len(self.gaussian)
        P = np.empty((n, k), dtype=np.float)
        for i in range(k):
            x_mean = self.gaussian[i]['x_mean']
            cov = self.gaussian[i]['cov']
            P[:, i] = stats.multivariate_normal.logpdf(X, mean=x_mean, cov=cov)
        return np.argmax(P, axis=1)


if __name__ == '__main__':
    training_data_size = 60000
    testing_data_size = 10000
    smoothing = 1e-2

    nnbc = NonNaiveBayesClassification()

    X, y = Loader.get_training_data('../data')
    print("Non-Naive Bayes classification training started with {} training data.".format(training_data_size))
    nnbc.train(X[:training_data_size], y[:training_data_size], smoothing)
    print("Non-Naive Bayes classification training finished.")

    print()

    Xt, yt = Loader.get_test_data('../data')
    print("Non-Naive Bayes evaluation started with {} testing data".format(testing_data_size))
    score = np.mean(nnbc.classify(Xt[:testing_data_size]) == yt[:testing_data_size])
    print("Evaluation finished with accuracy {:0.3f}%.".format(score*100))
