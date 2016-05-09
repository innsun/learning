# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def load_sentiment_data(url):
    data = pd.read_csv(url)
    y = data.iloc[:, 0]
    x = data.iloc[:, 1:]
    return x.values, y.values

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


class Bayes_matrix(object):
    def __init__(self):
        self.total_words = 0  # 所有单词总和
        self.words_train_num = 0  # 训练样本单词数
        self.words_num = np.zeros([2, 1], int)  # 所有分类单词0差的评论 1好的评论
        self.bad_words_num = []  # 按分类放坏数据
        self.good_words_num = []  # 按分类统计好数据
        self.all_words_num=[]
        # 加载数据

    def train_model(self, data, target):

        self.words_train_num = len(data[1,])
        bad_num=[]
        good_num=[]
        self.total_words =np.zeros([1,len(data[1,])])
        self.bad_words_num =np.zeros([1,len(data[1,])])
        self.good_words_num =np.zeros([1,len(data[1,])])
        for k in xrange(len(target)):  # len(target)):
            if target[k] == 0:
                self.words_num[0] += np.sum(data[k,])
                self.bad_words_num+=data[k,]
            else:
                self.words_num[1] += np.sum(data[k,])
                self.good_words_num+=data[k,]
        self.total_words = np.sum(self.words_num)



    def test_model(self, data, target):
          pvec=[]
          ptest=[]
          for m in xrange(len(target)):                       #从第一行到m行
             ptest.append(target[m])
             col_len = len(data[1,])                # 有多少列
             pbad=1
             pgood=1
             #print col_len
             count_bad=[]
             count_good=[]
             for n in xrange(col_len):            # 第一列到n列
                 if data[m,n]<>0:
                     count_bad.append(self.bad_words_num[:,n])
                     count_good.append(self.good_words_num[:,n])
             for i in xrange(len(count_bad)):
                 pbad*=float(count_bad[i]+1.0)/float(self.words_num[0]+self.words_train_num)
             pbad*=float(self.words_num[0])/float(self.total_words)
             for j in xrange(len(count_good)):
                 pgood*=float(count_good[j]+1.0)/float(self.words_num[1]+self.words_train_num)
             pgood*=float(self.words_num[1])/float(self.total_words)
             if(pbad>pgood):
                 pvec.append(0)
             else:
                pvec.append(1)
          tp=0
          tn=0
          fn=0
          fp=0
          for k in xrange(len(ptest)):
            if ptest[k]==pvec[k] and pvec[k]==1:
                tp+=1
            if ptest[k]==pvec[k] and pvec[k]==0:
                tn+=1
            if ptest[k]+pvec[k] ==1 and pvec[k]==0:
                fn+=1
            if ptest[k]+pvec[k]==1 and pvec[k]==1:
               fp+=1

          pre=float(tp)/(float(tp)+float(fp))*100
          recall=float(tp)/(float(tp)+float(fn))*100
          fscore=float(2*pre*recall)/float(pre+recall)
          npre=float(tn)/(float(tn)+float(fn))*100
          nrecall=float(tn)/(float(tn)+float(fp))*100
          nfscore=float(2*npre*nrecall)/float(npre+nrecall)
          print '正面的准确率:%.2f%%'%pre
          print '召回率:%.2f%%'%recall
          print 'f1-score:%.2f%%'%fscore
          print '负面的准确率:%.2f%%'%npre
          print '召回率:%.2f%%'%nrecall
          print 'f1-score:%.2f%%'%nfscore
          return ptest,pvec



bayes = Bayes_matrix()
train_url = 'F:/Python/comment_sentiment_v2/train_set.csv'
test_url = 'F:/Python/comment_sentiment_v2/test_set.csv'
data1, target1 = load_sentiment_data(train_url)
bayes.train_model(data1, target1)
data2, target2 = load_sentiment_data(test_url)
ptest1,pvec1=bayes.test_model(data2, target2)
conf_arr = np.zeros((2,2))
for i in xrange(0,len(ptest1)):
      current_correct = ptest1[i]
      current_guess =   pvec1[i]
      conf_arr[current_correct][current_guess] += 1.0
makeconf(conf_arr, '', "comment")

