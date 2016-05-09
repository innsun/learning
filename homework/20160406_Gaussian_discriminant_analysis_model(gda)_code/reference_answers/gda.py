# coding=utf-8
# author=李鹏飞
import numpy as np
import util


# Gaussian discriminant analysismodel
class GDA:
    def __init__(self, x, y, name="GDA"):
        self.x, self.y, self.name = x, y, name
        self.m, self.n = np.shape(x)
        self.phi, self.mu0, self.mu1, self.sigma, self.sigma_mul_inv = 0, [], [], [], []

    # 训练模型
    def train(self):
        num_y1 = float(sum(self.y))
        num_y0 = self.m - num_y1
        self.phi = num_y1 / self.m
        self.mu0 = np.dot(1 - self.y.T, self.x) / num_y0
        self.mu1 = np.dot(self.y.T, self.x) / num_y1
        sub = self.x - np.dot(1 - self.y, self.mu0) - np.dot(self.y, self.mu1)
        self.sigma = np.dot(sub.T, sub) / (self.m - 1)
        self.sigma_mul_inv = np.linalg.pinv(self.sigma)

    # 多维高斯分布
    def gaussian(self, x, mu, phi):
        sub = (x - mu).T
        return -0.5 * np.expand_dims(sum(sub * np.dot(self.sigma_mul_inv, sub)), axis=1) + np.log(phi)

    # 预测结果
    def predict(self, x):
        p0 = self.gaussian(x, self.mu0, 1 - self.phi)
        p1 = self.gaussian(x, self.mu1, self.phi)
        return np.array([map(lambda k: 1 if k > 0 else 0, p1 - p0)]).T


def main():
    train_url = "train_set.csv"
    test_url = "test_set.csv"
    x, y = util.load_sentiment_data(train_url)
    gda = GDA(x, y)
    gda.train()
    x, y = util.load_sentiment_data(test_url)
    y1 = gda.predict(x)
    conf_arr = util.statistic(y, y1)
    util.make_conf(conf_arr, gda.name, "comment")
    util.show_rate(conf_arr)

main()
