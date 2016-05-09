# coding=utf-8
import pandas as pd
import numpy as np
import utils
import random
import math

def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_prime(x):
    return x*(1-x)


class NeuNet(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.mini_batch_size=0.02

    def neu(self,x,y):
        m,n=x.shape
        np.random.seed(1)
        weights0 =2*np.random.random((n,100))-1
        weights1 =2*np.random.random((100,30))-1
        weights2 =2*np.random.random((30,10))-1
        error=0
        l1_error =[]
        l2_error =[]
        l3_error =[]
        for j in xrange(200):

           for i in xrange(m):
             layer0 = x[i]
             layer1 = sigmoid(np.dot(layer0,weights0))
             layer2 = sigmoid(np.dot(layer1,weights1))
             layer3 = sigmoid(np.dot(layer2,weights2))

             l3_error = y[i] - layer3
             l3_delta = l3_error*sigmoid_prime(layer3)
             l2_error = l3_delta.dot(weights2.T)
             l2_delta = l2_error*sigmoid_prime(layer2)
             l1_error = l2_delta.dot(weights1.T)
             l1_delta = l1_error *sigmoid_prime(layer1)
             weights2 +=np.dot(layer2.reshape(30,1),l3_delta.reshape(1,10))
             weights1 +=np.dot(layer1.reshape(100,1),l2_delta.reshape(1,30))
             weights0 +=np.dot(layer0.reshape(324,1),l1_delta.reshape(1,100))
           if (j% 50) == 0:
               print "Error:" + str(np.mean(np.abs(l2_error)))
        return weights0,weights1,weights2

    def  test(self,x,y,w0,w1,w2):
         m,n=y.shape
         result=np.zeros((m,n))
         for i in xrange(m):
             layer0 = x[i]
             layer1 = sigmoid(np.dot(layer0,w0))
             layer2 = sigmoid(np.dot(layer1,w1))
             layer3 = sigmoid(np.dot(layer2,w2))
             #print layer3
             for j in xrange(len(layer3)):
                 if float(layer3[j])>=float(1.0-self.mini_batch_size):
                     layer3[j]=1
                 else:
                     layer3[j]=0
             result[i]=layer3
         return result

train_url = "D:/numerical_v1/train_set.csv"
test_url = "D:/numerical_v1/test_set.csv"
x, y = utils.load_data(train_url)
test_x, test_y = utils.load_data(test_url)
net = NeuNet([len(x[0]),100, 30, 10])
#ya1=net.neu(x[0],y[0].T)
w0,w1,w2=net.neu(x,y)
p= net.test(test_x,test_y,w0,w1,w2)
t=0
for k in xrange(len(p)):
    t1=[];
    t2=[]
    for n in xrange(10):
       t1.append(int(p[k][n]))
       t2.append(test_y[k][n])
    if t1==t2:
        t+=1
r=float(t)/float(len(p))*100
print "准确率 %.2f%%"%r