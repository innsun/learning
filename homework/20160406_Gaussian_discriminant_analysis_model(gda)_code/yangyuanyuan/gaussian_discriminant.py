# coding=utf-8
import public
import numpy as np


class GaussianDiscriminant(object):
    def __init__(self):
        self.train_vector, self.train_label, self.test_vector, self.test_label = [], [], [], []
        self.count = {"pos": 0, "neg": 0, "all": 0}
        self.model = {"mu0": 0, "mu1": 0, "sigma": 0}

    def load_vectors(self, train_url, test_url):
        self.train_vector, self.train_label = public.load_data(train_url)
        self.test_vector, self.test_label = public.load_data(test_url)
        self.train_label,self.test_label = np.array([self.train_label]).T, np.array([self.test_label]).T

    def tarin_model(self):
        self.count["all"] = len(self.train_label)
        self.count["pos"] = sum(self.train_label)
        self.count["neg"] = self.count["all"] - self.count["pos"]
        mu0 = (np.dot(self.train_label.T, self.train_vector) + 0.0) / self.count["pos"]
        mu1 = (np.dot(1-self.train_label.T, self.train_vector) + 0.0) / self.count["neg"]
        a = self.train_vector - np.dot(self.train_label, mu0) - np.dot(1-self.train_label, mu1)
        sigma = np.dot(a.T, a) / self.count["all"]
        self.model["mu0"], self.model["mu1"], self.model["sigma"] = mu0, mu1, sigma

    def predict(self):
        a = self.test_vector - np.dot(np.ones([len(self.test_label), 1]), self.model["mu0"])
        b = self.test_vector - np.dot(np.ones([len(self.test_label), 1]), self.model["mu1"])
        c = np.linalg.pinv(self.model["sigma"])
        p0 = np.log(float(self.count["pos"])/self.count["all"]) + (-0.5 * np.dot(np.dot(a, c), a.T)).diagonal()
        p1 = np.log(float(self.count["neg"])/self.count["all"]) + (-0.5 * np.dot(np.dot(b, c), b.T)).diagonal()
        pre_y = map(lambda x:1 if x > 0 else 0, p0-p1)
        print pre_y
        public.analyse(pre_y, self.test_label.T[0])

gaussian = GaussianDiscriminant()
gaussian.load_vectors("E:/python_tmp/comment/comment_sentiment_v2/train_set.csv",
                   "E:/python_tmp/comment/comment_sentiment_v2/test_set.csv")
gaussian.tarin_model()
gaussian.predict()

