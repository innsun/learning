# coding=utf-8
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter

#从csv文件中加载数据
def load_sentiment_data(url):
    data = pd.read_csv(url)
    y = data.iloc[:, 0]
    x = data.iloc[:, 1:]
    return x.values, y.values


def makeconf(conf_arr, model, dataSet):
    # makes a confusion matrix plot when provided a matrix conf_arr
    # model and dataset just for pathname to save figure
    norm_conf = []
    for xi in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(xi, 0)
        for j in xi:
            tmp_arr.append(float(j)/float(a))
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
    plt.savefig("cm_" + dataSet + "_" + model + ".png")

def print_result(rs):
    precision = 1.0 * rs[2] / (rs[2] + rs[1])
    recall = 1.0 * rs[2] / (rs[2] + rs[0])
    print '正面准确率 : ' + str(recall) + ', 召回率 : ' + str(precision) + ', F1 : ' + str(2.0 * (precision * recall) / (precision + recall))
    precision = 1.0 * rs[-1.0] / (rs[-1.0] + rs[0])
    recall = 1.0 * rs[-1.0] / (rs[-1.0] + rs[1])
    print '负面准确率 : ' + str(recall) + ', 召回率 : ' + str(precision) + ', F1 : ' + str(2.0 * (precision * recall) / (precision + recall))

def classify_result(p_c0, p_c1, y):
    # 分类情况，+1表示分到1类，-1表示分到0类
    c = p_c1 - p_c0
    c /= np.abs(c)
    rs = Counter(np.nan_to_num(c) + y)
    # 构建混淆矩阵
    conf_arr = np.ones((2, 2))
    conf_arr[0, 0] = rs[-1.0]
    conf_arr[0, 1] = rs[1]
    conf_arr[1, 0] = rs[0]
    conf_arr[1, 1] = rs[2]
    makeconf(conf_arr, 'gaussian', "comment")
    print_result(rs)