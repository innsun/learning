# coding=utf-8
# author=李鹏飞
import numpy as np
import util


class NaiveBayes:
    # 伯努利模型
    BERNOULLI_MODEL = 0
    # 多项式模型
    POLYNOMIAL_MODEL = 1

    def __init__(self, x, y, data_type, mode=BERNOULLI_MODEL, name="naive_bayes"):
        self.x, self.y1, self.y2, self.data_type, self.name = x, y, y, data_type, name
        self.xm, self.xn = np.shape(x)
        self.tm, self.tn = np.shape(data_type)
        if self.tm != 1:
            raise AttributeError("数据类型必须为R(1*n)的向量")
        self.sample_probability, self.words_probability = [], []
        # 如果为伯努利模型，则平滑值取2
        if mode == NaiveBayes.BERNOULLI_MODEL:
            self.smooth_value = 2.0
        # 如果为多项式模型，则平滑值取词语类型数
        elif mode == NaiveBayes.POLYNOMIAL_MODEL:
            self.smooth_value = self.xn
        else:
            raise AttributeError("模型错误")

    # 横向增广矩阵
    def my_stack(self):
        self.y2 = util.transform(self.y1, self.data_type[0, 0])
        for i in range(1, self.tn):
            self.y2 = np.hstack((self.y2, util.transform(self.y1, self.data_type[0, i])))

    # 预测分类判别
    def judge(self, test_probability):
        m, n = np.shape(test_probability)
        c = np.zeros((m, 1))
        for i in range(0, m):
            max_i, max_j = util.max_index(np.array([test_probability[i, :]]))
            c[i, 0] = self.data_type[max_i, max_j]
        return c

    # 训练模型
    def train(self):
        # 按分类生成正负面矩阵
        self.my_stack()
        # 正负面情感概率
        self.sample_probability = (np.array([sum(self.y2)]) / self.xm).T
        # 情感条件词语数目
        words_num = np.dot(self.x.T, self.y2)
        # 情感条件词语概率
        self.words_probability = ((words_num + 1.0) / (np.array([sum(words_num)]) + self.smooth_value))

    # 预测结果
    def predict(self, x, y, data_set):
        # 测试集正负面情感预测概率
        test_probability = util.exp_mul(x, self.words_probability) * self.sample_probability.T
        # 统计混淆矩阵
        conf_arr = util.statistic(y, self.judge(test_probability), self.tn)
        # 绘制混淆矩阵
        util.make_conf(conf_arr, self.name, data_set)


# 主方法
def main():
    # 指定数据路径
    train_url = "F:/Python/comment_sentiment_v2/train_set.csv"
    test_url = "F:/Python/comment_sentiment_v2/test_set.csv"
    # 定义数据分类类型，可以适应多分类问题
    data_type = np.array([[0, 1]])
    # 加载训练集
    x, y = util.load_sentiment_data(train_url)
    bayes = NaiveBayes(x, y, data_type)
    # 训练模型
    bayes.train()
    # 验证测试集
    x, y = util.load_sentiment_data(test_url)
    bayes.predict(x, y, "comment")


main()
