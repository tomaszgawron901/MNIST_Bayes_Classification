from data import MNIST_Data_Loader as Loader
from bayesian_classification.Naive_Bayes_Classification import NaiveBayesClassification
from bayesian_classification.Non_Naive_Bayes_Classification import NonNaiveBayesClassification

import numpy as np

if __name__ == '__main__':
    X, y = Loader.get_training_data('../data')
    nbc = NaiveBayesClassification()
    training_data_size = 10000
    print("Naive Bayes classification training started with {} training data.".format(training_data_size))
    nbc.train(X[:training_data_size], y[:training_data_size])
    print("Naive Bayes classification training finished.")

    Xt, yt = Loader.get_test_data('../data')
    testing_data_size = 3000
    print("Naive Bayes evaluation started with {} testing data".format(testing_data_size))
    score = np.mean(nbc.classify(Xt[:testing_data_size]) == yt[:testing_data_size])
    print("Evaluation finished with accuracy {:0.3f}%.".format(score*100))

    print()

    nnbc = NonNaiveBayesClassification()
    training_data_size = 10000
    print("Non-Naive Bayes classification training started with {} training data.".format(training_data_size))
    nnbc.train(X[:training_data_size], y[:training_data_size])
    print("Non-Naive Bayes classification training finished.")

    testing_data_size = 3000
    print("Non-Naive Bayes evaluation started with {} testing data".format(testing_data_size))
    score = np.mean(nnbc.classify(Xt[:testing_data_size]) == yt[:testing_data_size])
    print("Evaluation finished with accuracy {:0.3f}%.".format(score*100))
