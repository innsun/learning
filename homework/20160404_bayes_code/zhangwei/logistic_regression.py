# coding=utf-8 ##以utf-8编码储存中文字符
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import util as cm
# logistic
# author 张维


# Hypothesis function
def h_theta(theta, data_area):
    temp_area = 1.0 / (1 + np.exp(-np.dot(data_area, theta)))
    return temp_area

# 训练数据 cost function
def training(train_url):
    x, z, y = cm.load_sentiment_data(train_url)
    y.shape = (len(y), 1)
    w = np.transpose(x)
    theta = np.ones((len(x[0, :]), 1))
    alpha = 0.05
    for i in range(0, 100):
        theta = theta - alpha * np.dot(w, (h_theta(theta, x) - y))
    return theta


# 启动方法
def main():
    train_url = "F:/Python/comment_sentiment_v2/train_set.csv"
    test_url = "F:/Python/comment_sentiment_v2/test_set.csv"
    theta = training(train_url)
    x, z, y = cm.load_sentiment_data(test_url)
    result_data = np.rint(h_theta(theta, x))
    cm.print_data(result_data, y)


main()
