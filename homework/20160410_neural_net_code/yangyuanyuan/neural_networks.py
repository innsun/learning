# coding=utf-8

import numpy as np
import public
import time


class NeuralNetworks(object):
    def __init__(self, layer, sizes, freq, alphy, epsilon):
        self.layer, self.sizes, self.freq = layer, sizes, freq  #层数, 每层大小
        self.thetas, self.train_vector, self.train_label = [], [], []
        self.alphy, self.epsilon = alphy, epsilon
        self.init_theta()

    def init_theta(self):
        for i in range(self.layer-1):
            self.thetas.append(np.random.random(size=(self.sizes[i+1], self.sizes[i]+1)) * self.epsilon * 2 - self.epsilon)

    def forward_propagation(self, theta, vector):
        a = [0] * self.layer
        a[0] = vector
        for i in range(1, self.layer):
            val = np.dot(theta[i-1], np.c_[np.ones([np.shape(a[i-1])[0], 1]), a[i-1]].T).T
            a[i] = public.sigmod(val)
        return a

    def back_propagation(self, theta, a):
        delta = [0] * self.layer
        delta[self.layer-1] = a[self.layer-1] - self.train_label
        for i in range(1, self.layer-1)[::-1]:
            v1 = np.dot(delta[i+1], theta[i])
            v2 = np.c_[np.ones([np.shape(a[i])[0], 1]), a[i]]
            d = v1 * v2 * (1-v2)
            delta[i] = d[:, 1:]
        return delta

    def gradient(self, theta):
        a = self.forward_propagation(theta, self.train_vector)
        delta = self.back_propagation(theta, a)
        temp_theta = [0] * len(theta)
        for i in range(self.layer-1):
            d = np.dot(delta[i+1].T, a[i])
            d = np.c_[map(sum, zip(*delta[i+1])), d]
            temp_theta[i] = theta[i] - self.alphy * d / len(self.train_label)
        return temp_theta

    def do_train(self, train_vector, train_label):
        self.train_vector, self.train_label = train_vector, train_label
        temp_theta = self.thetas
        for i in range(self.freq):
            temp_theta = self.gradient(temp_theta)
        self.thetas = temp_theta

    def predict(self, test_vector):
        p_max_index = self.forward_propagation(self.thetas, test_vector)[self.layer-1].argmax(axis=1)
        return p_max_index


def main():
    start = time.time()
    k = 10
    x, y = public.load_pics("E:/python_tmp/number", k)
    x1, y1, x2, y2 = public.get_dataset(x, y, len(x), 500, np.shape(x)[1], k)
    neural = NeuralNetworks(layer=3, sizes=[np.shape(x1)[1], 200, k], freq=1000, alphy=0.5, epsilon=0.1)
    neural.do_train(x1, y1)
    pre_y = neural.predict(x2)
    test_y = y2.argmax(axis=1)
    print "准确率:%s" % float(np.mean(test_y == pre_y))
    print "时间：%s" % (time.time() - start)

main()