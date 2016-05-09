# coding=utf-8
import numpy as np
import utils


class LogisticRegression(object):
    def __init__(self):
        self.alpha = 0.05
        self.theta = []

    # 设置参数
    def set_alpha(self, alpha):
        self.alpha = alpha

    # 训练模型
    def train_model(self, x, y):
        m, n = x.shape
        x = np.hstack((np.ones((m, 1)), x))
        self.theta = np.ones((n + 1, 1))
        for i in range(0, 200):
            self.theta -= np.dot(x.T, self.sigmoid(x, self.theta) - y) * self.alpha

    # 预测
    def predict(self, test_x):
        test_x = np.hstack((np.ones((test_x.shape[0], 1)), test_x))
        pre_y = np.array(np.array([map(round, self.sigmoid(test_x, self.theta))]).T, dtype="int")
        return pre_y

    # 逻辑回归的假设函数
    def sigmoid(self, x, theta):
        return 1.0 / (1 + np.exp(-np.dot(x, theta)))


def main():
    train_url = "e:/data/comment_sentiment/train_set.csv"
    test_url = "e:/data/comment_sentiment/test_set.csv"
    x, y = utils.load_data(train_url)
    test_x, test_y = utils.load_data(test_url)

    lr = LogisticRegression()
    lr.train_model(x, y)
    pre_y = lr.predict(test_x)
    utils.show_result(test_y, pre_y, "logistic_comment")

main()