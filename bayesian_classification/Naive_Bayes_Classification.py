from data import MNIST_Data_Loader as Loader

import numpy as np
from scipy import stats


class NaiveBayesClassification:
    def __init__(self):
        self.gaussian = []
        self.labels = np.empty(0)

    def train(self, X: np.ndarray, y: np.ndarray, smoothing=1.):
        self.labels = np.unique(y)
        for l in self.labels:
            divided_data = X[y == l]
            x_mean = np.mean(divided_data, axis=0)
            var = np.var(divided_data, axis=0) + smoothing
            self.gaussian.append({'x_mean': x_mean, 'var': var})

    def classify(self, X: np.ndarray):
        n, d = np.shape(X)
        k = len(self.gaussian)
        P = np.empty((n, k), dtype=np.float)
        for i in range(k):
            x_mean = self.gaussian[i]['x_mean']
            var = self.gaussian[i]['var']
            P[:, i] = stats.multivariate_normal.logpdf(X, mean=x_mean, cov=var)
        return np.argmax(P, axis=1)


if __name__ == '__main__':
    training_data_size = 60000
    testing_data_size = 10000
    smoothing = 1e-2

    nbc = NaiveBayesClassification()

    X, y = Loader.get_training_data('../data')
    print("Naive Bayes classification training started with {} training data.".format(training_data_size))
    nbc.train(X[:training_data_size], y[:training_data_size], smoothing)
    print("Naive Bayes classification training finished.")

    Xt, yt = Loader.get_test_data('../data')
    print("Naive Bayes evaluation started with {} testing data".format(testing_data_size))
    score = np.mean(nbc.classify(Xt[:testing_data_size]) == yt[:testing_data_size])
    print("Evaluation finished with accuracy {:0.3f}%.".format(score*100))
