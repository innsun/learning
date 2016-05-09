# coding=utf-8
# author=陈德利
import time
import numpy as np
import pandas as pd


class Layer(object):
    """ 神经网络中的单一层。
    Attributes:
        z:接收到的值
        a:激励值
        theta:权值
        error:误差
    """

    def __init__(self, index, alpha, input_feature_count, output_neuron_count):
        self.index, self.alpha = index, alpha
        input_feature_count += 1  # 加入常数项的theta
        self.theta = np.random.random_sample((output_neuron_count, input_feature_count)) - 0.5

    def forward_propagating(self, prev_layer):
        """ 前向传播。 """
        ro, co = prev_layer.a.shape
        x = np.hstack((np.ones((ro, 1)), prev_layer.a))
        self.z = np.dot(x, prev_layer.theta.T)
        self.a = 1 / (1 + np.exp(-self.z))
        return self.a

    def back_propagating(self, next_layer):
        """ 反向传播。 """
        self.error = np.dot(next_layer.error, self.theta[:, 1:]) * self.a * (1 - self.a)
        return self.error

    def gradient_descent(self, next_layer):
        """ 梯度下降。 """
        ro, co = self.a.shape
        x = np.hstack((np.ones((ro, 1)), self.a))
        cost_derivative = np.dot(x.T, next_layer.error) / x.size
        self.theta -= self.alpha * cost_derivative.T


class NeuralNetwork(object):
    """ 对整个神经网络的管理。
    Attributes:
        input_layer: 输入层
        hidden_layers: 隐藏层
        output_layer: 输出层
    """

    def __init__(self, hidden_layer_count, hidden_layer_neuron_count, alpha, train_times):
        """ 构建一个指定规模的神经网络。
        Args:
            hidden_layer_count: 隐藏层数
            hidden_layer_neuron_count: 每个隐藏层中神经元数
            alpha: 学习速率
            train_times: 训练次数
        """
        self.hidden_layer_count, self.hidden_layer_neuron_count = hidden_layer_count, hidden_layer_neuron_count
        self.alpha, self.train_times = alpha, train_times

    def train(self, x, y):
        """ 训练模型。 """
        xrow, xcol = x.shape
        yrow, ycol = y.shape
        # 定义输入层
        self.input_layer = Layer(1, self.alpha, xcol, self.hidden_layer_neuron_count)
        # 定义隐藏层
        self.hidden_layers = list()
        for i in range(1, self.hidden_layer_count):
            layer = Layer(i + 1, self.alpha, self.hidden_layer_neuron_count, self.hidden_layer_neuron_count)
            self.hidden_layers.append(layer)
        self.hidden_layers.append(Layer(self.hidden_layer_count + 1, self.alpha, self.hidden_layer_neuron_count, ycol))
        # 定义输出层
        self.output_layer = Layer(self.hidden_layer_count + 2, self.alpha, 0, 0)
        # 训练模型
        for i in range(0, self.train_times):
            self.forward_propagating(x)
            self.back_propagating(y)
            self.gradient_descent()

    def predict(self, x):
        """ 预测分类。"""
        self.forward_propagating(x)
        return self.output_layer.a

    def forward_propagating(self, x):
        """ 整个神经网络的一次前向传播过程。"""
        prev_layper = self.input_layer
        prev_layper.a = x
        for i in range(0, len(self.hidden_layers), 1):
            self.hidden_layers[i].forward_propagating(prev_layper)
            prev_layper = self.hidden_layers[i]
        self.output_layer.forward_propagating(prev_layper)

    def back_propagating(self, y):
        """ 整个神经网络的一次反向传播过程。"""
        self.output_layer.error = self.output_layer.a - y
        # TODO 简单的监视误差变化情况，待优化
        cost_sum = (self.output_layer.error * self.output_layer.error).sum()
        print ("整体误差：{}".format(cost_sum))
        next_layer = self.output_layer
        for i in range(len(self.hidden_layers) - 1, -1, -1):
            self.hidden_layers[i].back_propagating(next_layer)
            next_layer = self.hidden_layers[i]

    def gradient_descent(self):
        """整个神经网络的一次梯度下降过程。 """
        next_layer = self.output_layer
        for i in range(len(self.hidden_layers) - 1, -1, -1):
            self.hidden_layers[i].gradient_descent(next_layer)
            next_layer = self.hidden_layers[i]
        self.input_layer.gradient_descent(next_layer)


def load_data(url):
    """ 加载数据。 """
    data = pd.read_csv(url)
    y = data.iloc[:, 0:10]
    x = data.iloc[:, 10:]
    return x.values, y.values


def compare_result(expect, actual, cost_time):
    """ 查看结果。 """
    rs = expect.argmax(1) - actual.argmax(1)  # 比较期望值与实际结果中最大值的位置
    rs /= rs  # 位置相同时为0，否则为1。【基于这样一个前提：经测试发现，0/0时不会报错，结果是0】
    total = rs.size
    wrong = rs.sum()
    right = rs.size - wrong
    print ("正确率:{}, 总耗时：{}秒".format(1.0 * right / total, cost_time))


def main():
    # 记录开始时间
    begin = time.time()
    # 构建训练模型的实例
    neural_network = NeuralNetwork(hidden_layer_count=1, hidden_layer_neuron_count=100, alpha=1000, train_times=500)
    # 训练模型
    x, y = load_data("H:/lonton_a/workspace/numerical_v1/train_set.csv")
    neural_network.train(x, y)
    # 测试模型
    x, y = load_data("H:/lonton_a/workspace/numerical_v1/test_set.csv")
    actual = neural_network.predict(x)
    # 记录结束时间
    end = time.time()
    # 查看结果
    print ("输出层结果：\n{}".format(actual))
    compare_result(expect=y, actual=actual, cost_time=end - begin)


main()
