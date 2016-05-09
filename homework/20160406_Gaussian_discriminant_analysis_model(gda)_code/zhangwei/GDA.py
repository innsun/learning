# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import util
import math


# Gaussian discriminant analysismodel
# author 张维 2016.04.05
class GDA:
    def __init__(self):
        self.pi_0 = 0
        self.pi_1 = 0
        self.mu_0 = 0
        self.mu_1 = 0
        self.mu = 0
        self.cov = 0

    # 训练数据
    def training(self, train_x, train_y):
        y = train_y
        x = train_x
        y_len = len(y)
        validate_y = np.ones((y_len, 1))
        y.shape = (1, y_len)
        self.pi_0 = np.dot(y, validate_y) / y_len
        self.pi_1 = 1 - self.pi_0
        self.mu_0 = 1.0 * np.dot(1 - y, x) / np.dot(1 - y, validate_y)
        self.mu_1 = 1.0 * np.dot(y, x) / np.dot(y, validate_y)
        self.mu = x - (y.T * self.mu_1 + (1 - y).T * self.mu_0)
        self.cov = 1.0 * np.dot(self.mu.T, self.mu) / (y_len - 1)

    # 测试模型
    def test_result(self, test_y, test_x):
        c1 = np.linalg.pinv(self.cov)
        y = test_y
        x = test_x
        p_x_y_0 = np.ones((len(y), 1))
        p_x_y_1 = np.ones((len(y), 1))
        temp0 = x - self.mu_0
        temp1 = x - self.mu_1
        # 待向量化
        for i in range(0, len(y)):
            t0 = temp0[i, :]
            t1 = temp1[i, :]
            p_x_y_0[i] = (np.log(self.pi_0) - 0.5 * np.dot(t0, np.dot(t0, c1).T))[0][0]
            p_x_y_1[i] = (np.log(self.pi_1) - 0.5 * np.dot(t1, np.dot(t1, c1).T))[0][0]
        p = p_x_y_1 - p_x_y_0
        for i in range(0, len(p)):
            if p[i] < 0:
                p[i] = 0
            else:
                p[i] = 1
        return p, y


def main():
    train_url = "H:/deeplearning/comment_sentiment_v2/train_set.csv"
    test_url = "H:/deeplearning/comment_sentiment_v2/test_set.csv"
    train_x, train_y = util.load_sentiment_data(train_url)
    test_x, test_y = util.load_sentiment_data(test_url)
    gda = GDA()
    gda.training(train_x, train_y)
    test_y, y = gda.test_result(test_x, test_y)
    util.print_data(test_y, y, "comment_data", "GDA")


main()
