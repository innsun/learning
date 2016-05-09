# coding=utf-8
# author=李鹏飞
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys


# 加载数据
def load_sentiment_data(url):
    data = pd.read_csv(url)
    x = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    return x.values, np.array([y.values]).T


# 统计混淆矩阵的值
def statistic(y1, y2, num=2):
    m1, n1 = np.shape(y1)
    m2, n2 = np.shape(y2)
    if n1 != 1 or n2 != 1 or m1 != m2:
        raise AttributeError("矩阵不能比较")
    conf_arr = np.zeros((num, num))
    for i in range(0, m1):
        conf_arr[y1[i, 0]][y2[i, 0]] += 1.0
    return conf_arr


# 绘制彩色混淆矩阵
def make_conf(conf_arr, model, data_set):
    # makes a confusion matrix plot when provided a matrix conf_arr
    # model and data_set just for pathname to save figure
    norm_conf = []
    for i in conf_arr:
        tmp_arr = []
        a = sum(i, 0) + sys.float_info.min
        for j in i:
            tmp_arr.append(float(j) / float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')

    width = len(conf_arr)
    height = len(conf_arr[0])

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    indexs = '0123456789'
    plt.xticks(range(width), indexs[:width])
    plt.yticks(range(height), indexs[:height])
    # you can save the figure here with:
    plt.savefig("cm_" + data_set + "_" + model + ".png")


# 获取概率最大的下标值
def max_index(a):
    m, n = np.shape(a)
    max_num, max_i, max_j = a[0, 0], 0, 0
    for i in range(0, m):
        for j in range(0, n):
            if a[i, j] > max_num:
                max_i = i
                max_j = j
    return max_i, max_j


# 将指定的值转换成1，其余的值转换成0
def transform(y, num):
    m, n = np.shape(y)
    b = np.zeros((m, n))
    for i in range(0, m):
        for j in range(0, n):
            if y[i, j] == num:
                b[i, j] = 1
            else:
                b[i, j] = 0
    return b


# 定义指数乘法cij *= bkj ** aik
def exp_mul(a, b):
    b = b.T
    m, n = np.shape(b)
    c = np.array([np.prod(b[0, :] ** a, 1)]).T
    for i in range(1, m):
        c = np.hstack((c, np.array([np.prod(b[i, :] ** a, 1)]).T))
    return c


# 获取非零的个数
def nonzero_num(a):
    return np.sum(np.clip(a, 0, 1))


# 计算准确率，召回率和F1-score
def show_rate(conf_arr):
    tp, tn, fp, fn = conf_arr[1, 1], conf_arr[0, 0], conf_arr[0, 1], conf_arr[1, 0]
    # 计算准确率和召回率
    print "正面准确率：%.02f%%，召回率：%.02f%%，F1-score：%.02f%%" \
          % (tp / (tp + fp) * 100,
             tp / (tp + fn) * 100,
             2 * (tp / (tp + fn) * tp / (tp + fp)) / (tp / (tp + fn) + tp / (tp + fp)) * 100)
    print "负面准确率：%.02f%%，召回率：%.02f%%，F1-score：%.02f%%" \
          % (tn / (tn + fn) * 100,
             tn / (fp + tn) * 100,
             2 * (tn / (fp + tn) * tn / (tn + fn)) / (tn / (fp + tn) + tn / (tn + fn)) * 100)
