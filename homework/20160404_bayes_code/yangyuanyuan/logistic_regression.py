# coding=utf-8
import numpy as np
import public


class LogisticRegression(object):
    def __init__(self, alphy, freq, train_vector, train_label):
        self.alphy = alphy
        self.freq = freq
        self.train_vector, self.train_label = np.c_[np.ones((np.shape(train_vector)[0], 1)), train_vector], train_label
        self.theta = []

    def hypothesis(self, theta, x):
        y = 1/(1 + np.e**(-np.dot(x, theta)))
        return y.T

    def gradient(self, x, y, theta):
        h = self.hypothesis(theta, x) - y
        temptheta = theta - (np.dot(h, x) * self.alphy).T
        return temptheta

    def do_gradient(self):
        initial_theta = np.zeros((len(self.train_vector[0]), 1))
        for i in range(self.freq):
            initial_theta = self.gradient(self.train_vector, self.train_label, initial_theta)
        self.theta = initial_theta
        return initial_theta

    def do_predict(self, test_vector, test_label):
        pre_y = map(round, self.hypothesis(self.theta, np.c_[np.ones((np.shape(test_vector)[0], 1)), test_vector])[0])
        print pre_y
        public.analyse(pre_y, test_label)

a, b = public.load_data("F:/Python/comment_sentiment_v2/train_set.csv")
c, d = public.load_data("F:/Python/comment_sentiment_v2/test_set.csv")
logistic = LogisticRegression(0.05, 300, a, b)
initialTheta = logistic.do_gradient()
logistic.do_predict(c, d)