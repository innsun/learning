# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 读取数据
def load_sentiment_data(url):
    data = pd.read_csv(url)
    z = data.columns
    x = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    return x.values, z.values, y.values


def print_data(test_y, y):
    validate_x = np.ones((len(test_y), 1))
    validate_y = np.ones((1, len(test_y)))
    test_y = np.array(test_y, dtype="int")
    conf_arr = np.zeros((2, 2))
    test_y.shape = (1, len(test_y))
    f1 = 1.0 * np.dot(test_y, y) / np.dot(test_y, validate_x)
    b1 = 1.0 * np.dot(1 - test_y, 1 - y) / np.dot(1 - test_y, validate_x)
    f2 = 1.0 * np.dot(test_y, y) / np.dot(validate_y, y)
    b2 = 1.0 * np.dot(1 - test_y, 1 - y) / np.dot(validate_y, 1 - y)
    for i in range(0, len(test_y[0])):
        current_correct = y[i]
        current_guess = test_y[0][i]
        conf_arr[current_correct][current_guess] += 1.0
    makeconf(conf_arr, "test", "comment")
    print '正面准确率:', '%.2f' % f1, ' 正面召回率:', '%.2f' % f2, ' f1-score:', '%.2f' % (2 * f1 * f2 / (f1 + f2))
    print '负面准确率:', '%.2f' % b1, ' 负面召回率:', '%.2f' % b2, ' f1-score:', '%.2f' % (2 * b1 * b2 / (b1 + b2))

def makeconf(conf_arr, model, dataSet):
    # makes a confusion matrix plot when provided a matrix conf_arr
    # model and dataset just for pathname to save figure
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
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
