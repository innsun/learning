# coding=utf-8
from numpy import *
from util import *
def sigmoid(z):
    return 1.0/(1+np.exp(-z))
def train(x,y):
    m,n=np.shape(x)
    x=np.hstack((np.ones((m,1)),x))
    theta=np.ones((n+1))
    alpha=0.01
    maxIter=200
    #梯度下降
    for i in range(0,maxIter):
        A=np.dot(x,theta)
        E=sigmoid(A)-y
        theta=theta-alpha*np.dot(np.transpose(x),E)
    return theta
def predict(x,theta):
     m,n=np.shape(x)
     x=np.hstack((np.ones((m,1)),x))
     A=np.dot(x,theta)
     predict=sigmoid(A)
     pre_y=np.round(predict)
     return  np.int64(pre_y)
def testing():
    trainingSeturl = 'D:/DeepLearning/bayes/train_set.csv'
    testSeturl = 'D:/DeepLearning/bayes/test_set.csv'
    train_x, train_y = load_data(trainingSeturl)
    test_x, test_y = load_data(testSeturl)
    #{'alpha': 0.01, 'maxIter': 200, 'optimizeType': 'gradDescent'}
    theta=train(train_x,train_y)
    pre_y=predict(test_x,theta)
    analyse(pre_y, test_y)
    conf_arr = np.zeros((2, 2))
    for i in xrange(0, len(test_y)):
        current_correct = test_y[i]
        current_guess = pre_y[i]
        conf_arr[current_correct][current_guess] += 1.0
    makeconf(conf_arr, "LR", "comment")
testing()
