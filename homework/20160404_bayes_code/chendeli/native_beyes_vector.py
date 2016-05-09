# coding=utf-8
#! /usr/bin/python
from collections import Counter
import numpy as np
import util

#加载训练数据
train_url = "F:/Python/comment_sentiment_v2/train_set.csv"
x, y = util.load_sentiment_data(train_url)
# 每类发生的概率
fc = Counter(y)
pc = (np.vstack((fc[0], fc[1]))/[1, y.size * 1.0])[:,1]
#每类下各特征发生的概率
dict = {0:list(), 1:list()}
for i in range(0, y.size):
    dict[y[i]].append(x[i])
fdc0 = np.array(dict[0]).sum(0)
fdc1 = np.array(dict[1]).sum(0)
fdc = np.vstack((fdc0, fdc1))
pdc = (fdc+1.0)/np.array([fdc.sum(1) + 2.0]).T


#加载测试数据
train_url = "F:/Python/comment_sentiment_v2/test_set.csv"
x, y = util.load_sentiment_data(train_url)
#文本属于各分类的概率(省去了分母)
p0 = (pdc[0] ** x).prod(1)*pc[0]
p1 = (pdc[1] ** x).prod(1)*pc[1]
#分类情况，+1表示分到1类，-1表示分到0类
c = p1 - p0
c /= np.abs(c)
rs = Counter(np.nan_to_num(c) + y)
#打印出混淆矩阵
conf_arr = np.ones((2,2))
conf_arr[0,0] = rs[-1.0]
conf_arr[0,1] = rs[1]
conf_arr[1,0] = rs[0]
conf_arr[1,1] = rs[2]
util.makeconf(conf_arr, 'nativebeyes', "comment")
