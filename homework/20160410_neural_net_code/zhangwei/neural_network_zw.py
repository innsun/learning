# -*- coding: utf-8 -*-
# author 张维 2016.04.09

import time
import numpy as np
import util as u


class neural_network:
    def __init__(self, alpha=3, gradient_times=500):
        self.theta1 = []
        self.theta2 = []
        self.a1 = []
        self.a2 = []
        self.a3 = []
        self.alpha = alpha
        self.gradient_times = gradient_times

    def h_theta(self, theta, data_area):
        temp_area = 1.0 / (1 + np.exp(-np.dot(data_area, theta)))
        return temp_area

    def cost_function(self, a3, train_y, len_x_c, t):
        """ 输出代价函数 """
        cost = -1 / len_x_c * (train_y * np.log10(a3) + (1 - train_y) * np.log10(1 - a3)).sum()
        if t % 10 == 0:
            print("迭代次数：%d，代价函数值：%.2f" % (t, cost))

    def training(self, train_x, train_y):
        self.a1 = train_x
        len_x_c = len(self.a1[0, :])
        len_y_c = len(train_y[0, :])
        len_a = ((len_y_c + len_x_c) / 2)
        self.theta1 = np.random.random_sample((len_x_c, len_a))
        self.theta2 = np.random.random_sample((len_a, len_y_c))
        for i in range(0, self.gradient_times):
            self.a2 = self.h_theta(self.theta1, self.a1)
            self.a3 = self.h_theta(self.theta2, self.a2)
            deta_3 = self.a3 - train_y
            deta_2 = np.dot(deta_3, self.theta2.T) * self.a2 * (1 - self.a2)
            self.theta1 -= 1.0 / len_x_c * self.alpha * np.dot(self.a1.T, deta_2)
            self.theta2 -= 1.0 / len_x_c * self.alpha * np.dot(self.a2.T, deta_3)

    def test(self, test_x):
        self.a1 = test_x
        self.a2 = self.h_theta(self.theta1, self.a1)
        self.a3 = self.h_theta(self.theta2, self.a2)
        return self.a3

    def gredient_checking(self):
        """
        利用数值检验方法检验这些偏导数
        """
        pass


def main():
    train_url = "./train_set.csv"
    test_url = "./test_set.csv"
    time_0 = time.time()
    nnc = neural_network()
    pu = u.public_util()
    train_x, train_y = pu.load_img_data(train_url)
    test_x, test_y = pu.load_img_data(test_url)
    nnc.training(train_x, train_y)
    result_data = np.rint(nnc.test(test_x))
    time_1 = time.time()
    validate_y = 0
    for i in range(0, len(result_data)):
        if (result_data[i] == test_y[i]).all:
            validate_y += 1
    print "耗时：", time_1 - time_0, "正确率：", 1.0 * validate_y / len(test_y)


if __name__ == '__main__':
    main()
