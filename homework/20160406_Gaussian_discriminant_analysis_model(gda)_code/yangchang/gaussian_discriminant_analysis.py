# coding=utf-8
import numpy as np
import utils


class GDA(object):
    def __init__(self):
        self.phi = 0
        self.mu0, self.mu1 = 0, 0
        self.sigma = []

    # 训练模型
    def train_model(self, x, y):
        m, n = x.shape
        self.phi = float(y.sum()) / m
        self.mu0 = np.dot(1 - y.T, x) / float(m - y.sum())
        self.mu1 = np.dot(y.T, x) / float(y.sum())
        mu = x - (y * self.mu1 + (1 - y) * self.mu0)
        self.sigma = np.dot(mu.T, mu) / m

    # 预测
    def predict(self, test_x):
        inverse = np.linalg.pinv(self.sigma)
        temp0 = test_x - self.mu0
        temp1 = test_x - self.mu1
        p0 = np.log(1 - self.phi) + (-(np.dot(np.dot(temp0, inverse), temp0.T).diagonal()) / 2)
        p1 = np.log(self.phi) + (-(np.dot(np.dot(temp1, inverse), temp1.T).diagonal()) / 2)
        pre_y = np.array([map(lambda x: 1 if x > 0 else 0, p1 - p0)]).T
        return pre_y


def main():
    train_url = "train_set.csv"
    test_url = "test_set.csv"
    x, y = utils.load_data(train_url)
    test_x, test_y = utils.load_data(test_url)

    gda = GDA()
    gda.train_model(x, y)
    pre_y = gda.predict(test_x)
    utils.show_result(test_y, pre_y, "gda_comment")


main()
