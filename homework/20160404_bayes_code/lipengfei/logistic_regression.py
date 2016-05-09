# coding=utf-8
# author=李鹏飞
import numpy as np
import util


class Logistic:
    def __init__(self, x, y, alpha=0.003, train_times=100, name="logistic_regression"):
        m, n = np.shape(x)
        self.x, self.y = np.hstack((np.ones((m, 1)), x)), y
        self.theta = np.zeros((n + 1, 1))
        self.alpha, self.train_times, self.name = alpha, train_times, name

    # sigmoid函数
    def sigmoid(self, x):
        return 1 / (np.exp(-np.dot(x, self.theta)) + 1)

    # 训练模型
    def train(self):
        for i in range(0, self.train_times):
            self.theta -= np.dot(self.x.T, self.sigmoid(self.x) - self.y) * self.alpha

    # 预测结果
    def predict(self, x, y, data_set):
        m, n = np.shape(x)
        conf_arr = util.statistic(y, np.round(self.sigmoid(np.hstack((np.ones((m, 1)), x)))))
        util.make_conf(conf_arr, self.name, data_set)


def main():
    # 指定数据路径
    train_url = "F:/Python/comment_sentiment_v2/train_set.csv"
    test_url = "F:/Python/comment_sentiment_v2/test_set.csv"
    # 加载训练数据
    x, y = util.load_sentiment_data(train_url)
    logistic = Logistic(x, y)
    # 训练数据
    logistic.train()
    # 加载测试数据
    x, y = util.load_sentiment_data(test_url)
    # 预测结果
    logistic.predict(x, y, "comment")


main()
