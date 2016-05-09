# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pylab
import os


def load_data(url):
        data = pd.read_csv(url)
        y = data.iloc[:, 0]
        x = data.iloc[:, 1:]
        return x.values, y.values


def makeconf(conf_arr, model, data_set):
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
        res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, interpolation='nearest')

        width = len(conf_arr)
        height = len(conf_arr[0])

        for x in xrange(width):
            for y in xrange(height):
                ax.annotate(str(conf_arr[x][y]), xy=(y, x), horizontalalignment='center', verticalalignment='center')

        cb = fig.colorbar(res)
        indexs = '0123456789'
        plt.xticks(range(width), indexs[:width])
        plt.yticks(range(height), indexs[:height])
        # you can save the figure here with:
        plt.savefig("cm_" + data_set + "_" + model + ".png")


# 测试准确率和召回率
def analyse(res, score):
        tp, fn, fp, tn = .0, .0, .0, .0
        for i, val in enumerate(score):
            if res[i]:
                if val:
                    tp += 1
                else:
                    fp += 1
            else:
                if val:
                    fn += 1
                else:
                    tn += 1
        precision1, recall1, precision2, recall2 = tp/(tp + fp), tp/(tp + fn), tn/(fn + tn), tn/(fp + tn)
        print "正面的准确率：%.02f%%；\t召回率：%.02f%%；\tf1-score:%.02f%%" % (precision1 * 100, recall1 * 100,100 * 2 * precision1 * recall1 / (precision1 + recall1))
        print "负面的准确率：%.02f%%；\t召回率：%.02f%%；\tf1-score:%.02f%%" % (precision2 * 100, recall2 * 100,100 * 2 * precision2 * recall2 / (precision2 + recall2))
        conf_arr = np.zeros((2,2))
        for i in xrange(0, len(score)):
            current_correct = score[i]
            current_guess = res[i]
            conf_arr[current_correct][current_guess] += 1.0
        makeconf(conf_arr, "hybrid_matrix", "comment")


def sigmod(val):
    res = 1/(1 + np.exp(-val))
    return res


def load_pic(url):
    img = pylab.array(Image.open(url).convert("L"))
    img = np.round(img/255.0)
    img = img.reshape((1, np.shape(img)[0]*np.shape(img)[1]))
    return img[0]


def load_pics(url, n):
    files = os.listdir(url)
    x, y = [], np.zeros([len(files), n])
    for i, f in enumerate(files):
        if os.path.isfile(url + "/" + f):
            data = load_pic(url + "/" + f)
            x.append(data)
            y[i] = get_out(int(f[0]), n)
    return np.array(x), y


def get_out(i, n):
    l = [0] * n
    l[i] = 1
    return np.array([l])


# 获取训练集和测试集
def get_dataset(data_x, data_y, count, m, n, k):
    x = np.zeros((m, n), dtype="int")
    y = np.zeros((m, k), dtype="int")
    test_x = np.zeros((count - m, n), dtype="int")
    test_y = np.zeros((count - m, k), dtype="int")
    train_i, test_i = 0, 0
    nums = count / k
    train_len = nums * m / count
    test_len = nums - train_len
    for i in range(0, len(data_x), nums):
        x[train_i:train_i + train_len] = data_x[i:i + train_len]
        y[train_i:train_i + train_len] = data_y[i:i + train_len]
        test_x[test_i:test_i + test_len] = data_x[i + train_len:i + nums]
        test_y[test_i:test_i + test_len] = data_y[i + train_len:i + nums]
        train_i += train_len
        test_i += test_len
    return x, y, test_x, test_y