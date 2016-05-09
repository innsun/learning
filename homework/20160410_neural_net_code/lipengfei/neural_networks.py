# coding=utf-8
# author=李鹏飞
import time
import numpy as np
import util


class Layer(object):
    def __init__(self, num, next_num, lamb=0.1, alpha=0.3, epsilon=1.0):
        self.num = num + 1
        self.lamb = lamb
        self.alpha = alpha
        if next_num != 0:
            self.theta = np.random.rand(num + 1, next_num) * 2 * epsilon - epsilon
            self.tm, self.tn = np.shape(self.theta)
        else:
            self.theta = None
        self.a, self.delta = np.array([[]]), np.array([[]])

    def put_a(self, a):
        m, n = np.shape(a)
        self.a = np.hstack((np.ones((m, 1)), a))

    def forward_propagation(self):
        return util.sigmoid(np.dot(self.a, self.theta))

    def back_propagation(self, delta):
        if self.theta is None:
            self.delta = delta
        else:
            self.delta = (np.dot(delta, self.theta.T) * self.a * (1 - self.a))[:, 1:]

    def gradient_descent(self, delta, m):
        self.theta -= self.alpha / m * (np.dot(self.a.T, delta) + self.lamb * np.vstack((np.zeros((1, self.tn)),
                                                                                         self.theta[1:, :])))


class NeuralNetworks(object):
    def __init__(self, input_num, output_num, hidden_num=None, lamb=0.1, alpha=0.3, epsilon=1.0, times=200):
        """
        默认按照输入层和输出层的均值提供一个隐藏层
        如果不需要隐藏层，需要提供空数组[]
        """
        self.times, self.lamb, self.alpha, self.epsilon = times, lamb, alpha, epsilon
        if hidden_num is None:
            hidden_num = [(input_num + output_num) / 2]
        if len(hidden_num) == 0:
            self.layer = [Layer(input_num, output_num, lamb, alpha, epsilon)]
        else:
            length = len(hidden_num)
            self.layer = [Layer(input_num, hidden_num[0], lamb, alpha, epsilon)]
            for i in range(0, length - 1):
                self.layer.append(Layer(hidden_num[i], hidden_num[i + 1], epsilon, lamb, alpha))
            self.layer.append(Layer(hidden_num[length - 1], output_num, lamb, alpha, epsilon))
            # next_num = 0
            self.layer.append(Layer(output_num, 0, lamb, alpha, epsilon))
        self.layer_num = len(self.layer)
        self.m = 0

    def forward_propagation(self, x):
        self.layer[0].put_a(x)
        for i in range(1, self.layer_num):
            self.layer[i].put_a(self.layer[i - 1].forward_propagation())

    def back_propagation(self, y):
        self.layer[self.layer_num - 1].back_propagation(self.layer[self.layer_num - 1].a[:, 1:] - y)
        for i in range(self.layer_num - 2, 0, -1):
            self.layer[i].back_propagation(self.layer[i + 1].delta)

    def gradient_descent(self):
        for i in range(0, self.layer_num - 1):
            self.layer[i].gradient_descent(self.layer[i + 1].delta, self.m)

    def train(self, x, y):
        mx, nx = np.shape(x)
        my, ny = np.shape(y)
        if mx != my:
            raise AttributeError("x和y样本数不统一")
        self.m = mx
        for i in range(0, self.times):
            self.forward_propagation(x)
            self.back_propagation(y)
            self.gradient_descent()

    def predict(self, x):
        self.forward_propagation(x)
        return self.layer[self.layer_num - 1].a[:, 1:]


def main():
    first_time = time.time()
    train_url = "train_set.csv"
    test_url = "test_set.csv"
    x, y = util.load_sentiment_data(train_url)
    _, input_num = np.shape(x)
    _, output_num = np.shape(y)
    neural_networks = NeuralNetworks(input_num, output_num)
    neural_networks.train(x, y)
    x, y = util.load_sentiment_data(test_url)
    last_time = time.time()
    print "正确率：%d%%" % (util.accuracy_rate(y, neural_networks.predict(x)) * 100)
    print "花费时间：%d秒" % (last_time - first_time)

if __name__ == "__main__":
    main()
