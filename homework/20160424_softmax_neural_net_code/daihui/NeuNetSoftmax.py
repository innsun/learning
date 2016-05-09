# coding=utf-8
import pandas as pd
import numpy as np
import random
import math
from PIL import Image
from pylab import *
import time
import struct


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return x * (1 - x)


def softmax(theta, thetai, xi):
    mat_dot = np.dot(thetai, xi.T)
    numerator = np.exp(mat_dot)
    mat_dot1 = np.dot(theta, xi.T)
    denominator = np.exp(mat_dot1)
    denominator = np.sum(denominator, axis=0)
    p = numerator / denominator
    return p


def softmax_prime(x):
    return x * (1 - x)


class NeuNet(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.mini_batch_size = 0.06
        self.learnrate=0.055
    def neu(self, x, y, category):
        m, n = x.shape
        k = y.shape[1]  # 为分类数
        np.random.seed(1)
        weights0 = []
        weights1 = []
        weights0 = 2 * np.random.random((n, 70)) - 1
        weights1  =2 * np.random.random((category,70)) - 1
        error = 0.0
        l1_error = []
        l2_error = []
        l3_error = []
        for j in xrange(20):

            for i in xrange(m):
                getnum = 0
                positon = np.argmax(y[i])

                layer0 = x[i]
                layer1 = sigmoid(np.dot(layer0, weights0))
                #print layer1.shape
                layer2 = np.zeros((category,1))
                for label in range(category):
                        weights1i = weights1[label,:]
                        p = softmax(weights1, weights1i, layer1)
                        layer2[label]=p
                #print layer2

                layer2=layer2.T
                l2_error = y[i] - layer2
                #print l2_error
                if (1.0-layer2[0,positon])>0.03:
                  l2_delta = l2_error * softmax_prime(layer2)
                  l1_error = l2_delta.dot(weights1)
                  l1_delta = l1_error * sigmoid_prime(layer1)
                  weights1 += self.learnrate*np.dot(l2_delta.reshape(1, k).T, layer1.reshape(70, 1).T)
                  weights0 += self.learnrate*np.dot(layer0.reshape(n, 1), l1_delta.reshape(1, 70))
                error = np.mean(np.abs(l2_error))

            if (j % 2) == 0:
                print "第" + str(j) + "轮 Error:" + str(error)
        return weights0, weights1

    def test(self, x, y, w0, w1, catgory):
        count = 0
        m, n = x.shape
        for i in xrange(m):
            prelist = []
            layer0 = x[i]
            layer1 = sigmoid(np.dot(layer0, w0))
            for label in range(category):
                thetai = w1[label, :]
                prob = softmax(w0, thetai, layer1)
                prelist.append(prob)

            prep = np.argmax(prelist)
            resp = np.argmax(y[i])
            if prep == resp:
                count += 1
        r = float(count) / float(x.shape[0]) * 100
        print "准确率 %.2f%%" % r


def load_mnist(url, is_image=True):
    with open(url, "rb") as f:
        buffers = f.read()
        type = ">IIII" if is_image else ">II"  # 读取文件头：4个或2个unsinged int32
        head = struct.unpack_from(type, buffers, 0)
        offset = struct.calcsize(type)
        num = head[1]
        fetures = head[2] * head[3] if is_image else 1
        bits = num * fetures
        bits_string = '>' + str(bits) + 'B'  # 读取文件内容：bits个unsigned byte

        data = struct.unpack_from(bits_string, buffers, offset)
        if is_image:
            data = map(lambda i: 1 if i > 100 else 0, data)  # 图片像素值处理，可根据自己需要调整
        data = np.reshape(data, [num, fetures])
        print "load minist finished：" + url
        return data


def get_classify(x, y, num):
    new_x, new_y = [], []
    for i, val in enumerate(y):
        if val < num:
            new_x.append(x[i])
            new_y.append(val)
    temp = np.zeros((len(new_y), num))
    for i, val in enumerate(new_y):
        temp[i][val] = 1  # 将0~9的数字映射到长度为10的数组上
        new_y = temp
    return np.array(new_x), new_y


url = "d:/mnist/"
start = time.time()
train_x_url = url + "train-images.idx3-ubyte"
train_y_url = url + "train-labels.idx1-ubyte"
test_x_url = url + "t10k-images.idx3-ubyte"
test_y_url = url + "t10k-labels.idx1-ubyte"
category = 10  # 只对012做数字识别
x = load_mnist(train_x_url)
y = load_mnist(train_y_url, is_image=False)
x, y = get_classify(x, y, category)
test_x = load_mnist(test_x_url)
test_y = load_mnist(test_y_url, is_image=False)
test_x, test_y = get_classify(test_x, test_y, category)

net = NeuNet([len(x[0]), 5, 30, 10])
# ya1=net.neu(x[0],y[0].T)
w0, w1 = net.neu(x, y, category)
net.test(test_x, test_y, w0, w1, category)


print("执行时间：%.3f秒" % (time.time() - start))
