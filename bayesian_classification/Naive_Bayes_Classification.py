import numpy as np
from scipy import stats


class NaiveBayesClassification:
    def __init__(self):
        self.gaussian = []
        self.labels = np.empty(0)

    def train(self, X: np.ndarray, y: np.ndarray, smoothing=0.001):
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

