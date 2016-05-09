# coding=utf-8
#!/usr/bin/python
from collections import Counter
import numpy as np
import util

#加载训练数据
train_url = "F:/Python/comment_sentiment_v2/train_set.csv"
x, y = util.load_sentiment_data(train_url)
# 初始化theta
ro,co = x.shape
theta = np.ones(co + 1)
# theta[0] = -1 #梯度下降多次后发现较趋近于-1，为减少迭代次数，初始即为-1
#重构x
x = np.hstack((np.ones((ro, 1)), x))
#设定学习速率
alpha = 0.002
#梯度下降
for i in range(0, 250):#TODO 对于何时结束还没有给出准确判断
    hx = np.dot(x, theta.T)
    gz = 1/(1 + np.e**(-hx))
    dTheta = np.array([gz-y]).T*x
    deltaTheta = np.sum(dTheta, 0)
    print  str(i) + ':' + str(np.sum(deltaTheta)) + str(theta[:2]) #观察斜率、前两项theta
    theta -= alpha * deltaTheta


#加载测试数据
train_url = "F:/Python/comment_sentiment_v2/test_set.csv"
x, y = util.load_sentiment_data(train_url)
#重构x
ro,co = x.shape
x = np.hstack((np.ones((ro, 1)), x))
#分类情况
hx = np.dot(x, theta.T)
gz = 1/(1 + np.e**(-hx))
c = gz - 0.5
c /= np.abs(c)
rs = Counter(np.nan_to_num(c) + y)
#打印出混淆矩阵
conf_arr = np.ones((2,2))
conf_arr[0,0] = rs[-1.0]
conf_arr[0,1] = rs[1]
conf_arr[1,0] = rs[0]
conf_arr[1,1] = rs[2]
util.makeconf(conf_arr, 'logistic', "comment")