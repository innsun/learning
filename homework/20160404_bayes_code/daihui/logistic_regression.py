# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



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


class LogisticRegression(object):
    def __init__(self):
        self.all_words_num=[]
        # 加载数据

    def train_model(self, data, target):
         self.all_words_num=map(sum,data)



    def sigmoid(self,inX):     # 逻辑回归
        return 1.0/(1+np.exp(-inX))

    def gradAscent(self,data,target):#梯度算法
        datamatrix = np.mat(data)
        targetmat = np.mat(target).transpose()
        m,n = np.shape(datamatrix)
        alpha = 0.05
        maxCycles =50
        thelamda=0.1
        weights = np.ones((n,1))
        for k in xrange(maxCycles):
            h = self.sigmoid(datamatrix*weights)
            error = (h-targetmat)
            weights = weights - alpha *( datamatrix.transpose()* error+thelamda*weights)
        return weights


    def test_logmodel(self, data, target,weights):
         pvec=[]
         ptest=[]
         for m in xrange(len(target)):
            ptest.append(target[m])
            col_len = len(data[1,])
            xx=np.zeros(col_len)
            yy=0
            for n in xrange(col_len):#第列
                if data[m,n]<>0:
                     xx[n] = float(self.all_words_num[n])
                     yy+=float(weights[n]*xx[n])

            if yy>0.5:
                pvec.append(1)
            else:
                pvec.append(0)
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


logist =LogisticRegression()
train_url = 'F:/Python/comment_sentiment_v2/train_set.csv'
test_url = 'F:/Python/comment_sentiment_v2/test_set.csv'
data1, target1 = load_sentiment_data(train_url)
weights1=logist.gradAscent(data1,target1)
logist.train_model(data1, target1)
data2, target2 = load_sentiment_data(test_url)
ptest1,pvec1=logist.test_logmodel(data2,target2,weights1)  #调用逻辑回归
conf_arr = np.zeros((2,2))
for i in xrange(0,len(ptest1)):
      current_correct = ptest1[i]
      current_guess =   pvec1[i]
      conf_arr[current_correct][current_guess] += 1.0
makeconf(conf_arr, '', "comment")
