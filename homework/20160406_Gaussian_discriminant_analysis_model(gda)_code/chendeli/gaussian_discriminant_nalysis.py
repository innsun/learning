# coding=utf-8
# !/usr/bin/python
import numpy as np
import util

# 加载训练数据
train_url = "H:/lonton_a/workspace/native beyes vector/set02/train_set.csv"
x, y = util.load_sentiment_data(train_url)
phi1 = 1.0 * y.sum() / y.size
phi0 = 1 - phi1
# 样本分类
dict = {0: list(), 1: list()}
for i in range(0, y.size):
    dict[y[i]].append(x[i])
x_c0 = np.array(dict[0])
x_c1 = np.array(dict[1])
# 计算mu
mu0 = 1.0 * x_c0.sum(0) / x_c0.shape[0]
mu1 = 1.0 * x_c1.sum(0) / x_c1.shape[0]
# 计算sigma,并预先处理好sigma的乘法逆元
deleta_c = np.vstack((x_c0 - mu0, x_c1 - mu1))
sigma = np.dot(deleta_c.T, deleta_c) / y.size
sigma_pinv = np.linalg.pinv(sigma)


def compare_value(sigma_pinv, mu, x):
    x_mu = x - mu
    return -(np.dot(x_mu, sigma_pinv) * x_mu).sum(1)

# 加载测试数据
train_url = "H:/lonton_a/workspace/native beyes vector/set02/test_set.csv"
x, y = util.load_sentiment_data(train_url)
p_c0 = compare_value(sigma_pinv, mu0, x) * phi0
p_c1 = compare_value(sigma_pinv, mu1, x) * phi1
# 打印出结果
util.classify_result(p_c0, p_c1,y)
