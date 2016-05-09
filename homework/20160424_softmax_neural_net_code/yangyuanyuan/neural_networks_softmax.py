# coding=utf-8

import time
import numpy as np
import public
import pca


class NeuralNetworks(object):
    def __init__(self, layer=3, sizes=[10, 200, 10], alphy=0.5, epsilon=0.1):
        self.layer, self.sizes, self.alphy= layer, sizes, alphy  #层数, 每层大小
        self.thetas = []
        self.init_theta(epsilon)

    def init_theta(self, epsilon):
        for i in range(self.layer-1):
            self.thetas.append(np.random.random(size=(self.sizes[i+1], self.sizes[i]+1)) * epsilon * 2 - epsilon)

    def forward_propagation(self, theta, x):
        a = [0] * self.layer
        a[0] = x
        for i in range(1, self.layer-1):
            val = np.dot(theta[i-1], np.c_[np.ones([np.shape(a[i-1])[0], 1]), a[i-1]].T).T
            a[i] = public.sigmod(val)
        val1 = np.dot(theta[self.layer-2], np.c_[np.ones([np.shape(a[self.layer-2])[0], 1]), a[self.layer-2]].T).T
        val2 = np.exp(val1)
        a[self.layer-1] = val2 / np.dot(np.array([val2.sum(axis=1)]).T, np.ones([1, np.shape(val2)[1]]))
        return a

    def back_propagation(self, theta, a, y):
        delta = [0] * self.layer
        v1 = np.dot(y, theta[self.layer-2]) - np.dot(a[self.layer-1], theta[self.layer-2])
        v2 = np.c_[np.ones([np.shape(a[self.layer-2])[0], 1]), a[self.layer-2]]
        delta[self.layer-2] = (-v1 * v2 * (1-v2))[:, 1:]
        for i in range(1, self.layer-2)[::-1]:
            v1 = np.dot(delta[i+1], theta[i])
            v2 = np.c_[np.ones([np.shape(a[i])[0], 1]), a[i]]
            delta[i] = (v1 * v2 * (1-v2))[:, 1:]
        return delta

    def gradient(self, theta, x, y):
        start = time.time()
        a = self.forward_propagation(theta, x)
        print "训练1时间：%s" % (time.time() - start)
        delta = self.back_propagation(theta, a, y)
        print "训练2时间：%s" % (time.time() - start)
        temp_theta = [0] * len(theta)
        for i in range(self.layer-2):
            d = np.dot(delta[i+1].T, np.c_[np.ones([np.shape(a[i])[0], 1]), a[i]])
            temp_theta[i] = theta[i] - self.alphy * d / len(y)
        d = np.dot((a[self.layer-1] - y).T, np.c_[np.ones([np.shape(a[self.layer-2])[0], 1]), a[self.layer-2]])
        temp_theta[self.layer-2] = theta[self.layer-2] - self.alphy * d / len(y)
        print "训练3时间：%s" % (time.time() - start)
        return temp_theta

    def do_train(self, x, y, freq=100):
        start = time.time()
        temp_theta = self.thetas
        for i in range(freq):
            temp_theta = self.gradient(temp_theta, x, y)
        self.thetas = temp_theta
        print "训练时间：%s" % (time.time() - start)

    def predict(self, test_vector):
        p_max_index = self.forward_propagation(self.thetas, test_vector)[self.layer-1].argmax(axis=1)
        return p_max_index


def main1():
    start = time.time()
    k = 10
    url = "E:/python_tmp/mnist/"
    train_x_url = url + "train-images.idx3-ubyte"
    train_y_url = url + "train-labels.idx1-ubyte"
    test_x_url = url + "t10k-images.idx3-ubyte"
    test_y_url = url + "t10k-labels.idx1-ubyte"
    x2 = public.load_mnist(test_x_url)
    x2 = pca.get_pca(x2, 500)
    y2 = public.load_mnist(test_y_url, is_image=False)
    y2 = public.types_change(y2, k)

    x1 = public.load_mnist(train_x_url)
    x1 = pca.get_pca(x1, 500)
    y1 = public.load_mnist(train_y_url, is_image=False)
    y1 = public.types_change(y1, k)

    print "数据准备时间：%s" % (time.time() - start)
    neural = NeuralNetworks(layer=3, sizes=[np.shape(x1)[1], 100, k], alphy=2, epsilon=0.1)
    neural.do_train(x1, y1, freq=200)
    pre_y = neural.predict(x2)
    test_y = y2.argmax(axis=1)
    print "准确率:%s" % float(np.mean(test_y == pre_y))
    print "时间：%s" % (time.time() - start)


def main():
    start = time.time()
    k = 10
    x, y = public.load_pics("E:/python_tmp/number", k)
    x1, y1, x2, y2 = public.get_dataset(x, y, len(x), 500, np.shape(x)[1], k)
    neural = NeuralNetworks(layer=3, sizes=[np.shape(x1)[1], 200, k], alphy=0.5, epsilon=0.1)
    neural.do_train(x1, y1, freq=1000)
    pre_y = neural.predict(x2)
    test_y = y2.argmax(axis=1)
    print "准确率:%s" % float(np.mean(test_y == pre_y))
    print "时间：%s" % (time.time() - start)

main1()