# coding=utf-8
# author 阳畅
import time
import numpy as np
import utils


class NeuralNetworks(object):
    def __init__(self, alpha=0.5, unit=200, hidden_layer=1):
        self.theta1 = []
        self.theta2 = []
        self.alpha = alpha
        self.unit = unit
        self.hidden_layer = hidden_layer

    def train_model(self, x, y):
        """ 训练模型 """
        m, n = x.shape
        k = y.shape[1]
        s1, s2, s3 = n, 200, k
        self.theta1 = np.random.random_sample((s1, s2)) * 0.1
        self.theta2 = np.random.random_sample((s2, s3)) * 0.1
        for t in range(1, 501):
            a1 = x
            a2 = utils.sigmoid(np.dot(a1, self.theta1))
            a3 = utils.sigmoid(np.dot(a2, self.theta2))
            d3 = a3 - y
            d2 = np.dot(d3, self.theta2.T) * a2 * (1 - a2)
            deta2 = np.dot(a2.T, d3)
            deta1 = np.dot(a1.T, d2)
            self.theta1 -= self.alpha * deta1 / m
            self.theta2 -= self.alpha * deta2 / m

    def predict(self, test_x):
        """ 预测 """
        m = test_x.shape[0]
        pre_y = np.zeros((m, 10))
        for i in range(0, m):
            a1 = test_x[i]
            a2 = utils.sigmoid(np.dot(a1, self.theta1))
            h = utils.sigmoid(np.dot(a2, self.theta2))
            pre_y[i][np.argmax(h)] = 1
        return pre_y

    def _show_cost(self, h, y, m, t):
        """ 输出代价函数 """
        cost = -1 / m * (y * np.log10(h) + (1 - y) * np.log10(1 - h)).sum()
        if t % 10 == 0:
            print("迭代次数：%d，代价函数值：%.2f" % (t, cost))


def main():
    start = time.time()
    train_url = "E:/lonton_a/data/numerical_v1/train_set.csv"
    test_url = "E:/lonton_a/data/numerical_v1/test_set.csv"
    x, y = utils.load_data(train_url, y_len=10)
    test_x, test_y = utils.load_data(test_url, y_len=10)

    networks = NeuralNetworks()
    networks.train_model(x, y)
    pre_y = networks.predict(test_x)
    print("执行时间：%.3f秒" % (time.time() - start))
    print("准确率：%.1f%%" % ((np.mean(test_y == pre_y) * 100)))


if __name__ == '__main__':
    main()
