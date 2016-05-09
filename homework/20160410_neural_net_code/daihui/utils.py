# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 加载数据
def load_data(url):
    data = pd.read_csv(url)
    y = data.iloc[:, 0:10]
    x = data.iloc[:, 10:]
    return x.values, y.values


# 展示结果
def show_result(test_y, pre_y, name):
    # makes a confusion matrix plot when provided a matrix conf_arr
    conf_arr = np.zeros((2, 2))
    test_y = test_y[:, 0]
    pre_y = pre_y[:, 0]
    for i in xrange(0, len(test_y)):
        current_correct = test_y[i]
        current_guess = pre_y[i]
        conf_arr[current_correct][current_guess] += 1.0
    print_result(conf_arr)

    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
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
    plt.savefig("cm_" + name + ".png")


# 输出预测结果
def print_result(conf_arr):
    print conf_arr
    precision1 = conf_arr[1][1] / (conf_arr[0][1] + conf_arr[1][1])
    recall1 = conf_arr[1][1] / (conf_arr[1][0] + conf_arr[1][1])
    positive_f1 = 2 * precision1 * recall1 / (precision1 + recall1)
    print ("正面准确率：%.4f" % precision1 + "，召回率：%.4f" % recall1 + "，F1：%.4f" % positive_f1)

    precision2 = conf_arr[0][0] / (conf_arr[1][0] + conf_arr[0][0])
    recall2 = conf_arr[0][0] / (conf_arr[0][1] + conf_arr[0][0])
    positive_f2 = 2 * precision2 * recall2 / (precision2 + recall2)
    print ("负面准确率：%.4f" % precision2 + "，召回率：%.4f" % recall2 + "，F1：%.4f" % positive_f2)
