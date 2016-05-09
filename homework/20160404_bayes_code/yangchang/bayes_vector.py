# coding=utf-8
import numpy as np
import utils


class Bayes(object):
    def __init__(self):
        self.good_count = 0
        self.bad_count = 0
        self.good_x = []
        self.bad_x = []

    # 训练模型
    def train_model(self, x, y):
        self.good_count = y.sum()
        self.bad_count = y.size - self.good_count
        self.good_x = np.dot(y.T, x)
        self.bad_x = np.dot(1 - y.T, x)

    # 预测
    def predict(self, x):
        pre = []
        total = self.good_count + self.bad_count
        for xi in x:
            p0, p1 = 1.0, 1.0
            for i, val in enumerate(xi):
                if val > 0:
                    p1 *= float(self.good_x[0][i] + 1) / (self.good_count + 2)
                    p0 *= float(self.bad_x[0][i] + 1) / (self.bad_count + 2)
            p1 *= self.good_count / float(total)
            p0 *= self.bad_count / float(total)
            pre.append(int(p1 - p0 > 0))
        return pre


def main():
    train_url = "e:/data/comment_sentiment/train_set.csv"
    test_url = "e:/data/comment_sentiment/test_set.csv"
    train_x, train_y = utils.load_data(train_url)
    test_x, test_y = utils.load_data(test_url)

    bayes = Bayes()
    bayes.train_model(train_x, train_y)
    pre_y = bayes.predict(test_x)
    utils.show_result(test_y, np.array([pre_y]).T, "bayes_comment")


main()
