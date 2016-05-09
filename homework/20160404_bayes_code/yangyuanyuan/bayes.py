# coding=utf-8
import numpy as np
import public


class Bayes(object):
    def __init__(self):
        self.dic = []
        self.count = {"pos": 0, "neg": 0, "all": 0}
        self.model = []
        self.train_vector, self.train_label, self.test_vector, self.test_label = [], [], [], []

    def train_model(self, data):
        self.create_dic(data)
        self.train_vector = self.get_vectors(data)
        self.train_label = np.array(data.score)
        self.create_model()

    def get_vector(self, data):
        v = np.zeros(len(self.dic))
        word_seg = list(jieba.cut(data))
        for i in range(len(word_seg)):
            if word_seg[i] in self.dic:
                index = self.dic.index(word_seg[i])
                v[index] = 1
        return v

    def get_vectors(self, data):
        vectors = np.zeros((len(data.score), len(self.dic)))
        for i, val in enumerate(data.content):
            v = self.get_vector(val)
            vectors[i] = v
        return vectors

    def load_vectors(self, train_url, test_url):
        self.train_vector, self.train_label = public.load_data(train_url)
        self.test_vector, self.test_label = public.load_data(test_url)

    def create_dic(self, data):
        for i in range(len(data.content)):
            seg_list = pseg.cut(data.content[i])
            for word, flag in seg_list:
                if word not in self.dic:
                    self.dic.append(word)

    def create_model(self):
        self.count["all"] = len(self.train_vector)
        self.count["pos"] = sum(self.train_label)
        self.count["neg"] = self.count["all"] - self.count["pos"]
        self.model = np.zeros((np.shape(self.train_vector)[1], 2))
        self.model[:, 0] = (np.dot(self.train_label.T, self.train_vector) + 1.0) / (self.count["pos"] + 2)
        self.model[:, 1] = (np.dot(1 - self.train_label.T, self.train_vector) + 1.0) / (self.count["neg"] + 2)

    def do_predict(self):
        res = []
        for vector in self.test_vector:
            p1 = float(self.count["pos"] + 1)/(self.count["all"] + 2)
            p2 = float(self.count["neg"] + 1)/(self.count["all"] + 2)
            for i in range(len(vector)):
                if vector[i] == 1:
                    p1 = p1 * self.model[i, 0]
                    p2 = p2 * self.model[i, 1]
            print "pos:%s; \t neg:%s;\t 结果:%s" % (p1, p2, p1 >= p2)
            r = 0
            if p1 >= p2:
                r = 1
            res.append(r)
        print res
        public.analyse(res, self.test_label)

    def predict_some(self, data):
        self.test_vector = self.get_vectors(data)
        self.test_label = np.array(data.score)
        self.do_predict()

bayes = Bayes()
bayes.load_vectors("F:/Python/comment_sentiment_v2/train_set.csv",
                   "F:/Python/comment_sentiment_v2/test_set.csv")
bayes.create_model()
bayes.do_predict()