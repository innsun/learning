# coding=utf-8
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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



