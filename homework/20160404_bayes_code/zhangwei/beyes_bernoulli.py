# coding=utf-8 ##以utf-8编码储存中文字符
import pandas as pd
import numpy as np
import util as cm

# polynomial beyes solve text classify problem 基于伯努利模型 二项式分布原理
# author 张维

# 测试数据
def test_result(test_path, train_data):
    df = train_data["df"]
    db = train_data["db"]
    f = train_data["fn"]
    b = train_data["bn"]
    train_url = test_path
    data = pd.read_csv(train_url)
    x = data.iloc[0:, 1:].values
    y = data.iloc[:, 0]
    test_y = []
    for j in range(0, len(x)):
        one = 1.0
        two = 1.0
        for i in range(0, len(df)):
            if x[j, i] == 1:
                one *= df[i]
                two *= db[i]
        one *= 1.0 * f / (f + b)
        two *= 1.0 * b / (f + b)
        if one > two:
            test_y.insert(j, 1)
        else:
            test_y.insert(j, 0)
    return test_y, y

# 训练数据
def training(train_url):
    train_data = {}
    train_url = train_url
    data = pd.read_csv(train_url)
    y = np.array(data.iloc[:, 0])
    x = np.array(data.iloc[:, 1:])
    xt = np.transpose(x)
    yt = 1 - y
    # 利用矩阵乘法计算每个词在多少个正面与负面文档中出现过
    yt = np.array(yt)
    y.shape = (len(y), 1)
    y = np.transpose(y)
    yt.shape = (len(yt), 1)
    yt = np.transpose(yt)
    y1 = np.ones((1, len(y[0])))
    train_data["fn"] = np.dot(y[0], y1[0])
    train_data["bn"] = len(data.iloc[:, 0]) - train_data["fn"]
    train_data["df"] = 1.0 * (np.dot(xt, y[0]) + 1) / (train_data["fn"] + 2)
    train_data["db"] = 1.0 * (np.dot(xt, yt[0]) + 1) / (train_data["bn"] + 2)
    return train_data


# 启动方法
def main():
    train_url = "F:/Python/comment_sentiment_v2/train_set.csv"
    test_url = "F:/Python/comment_sentiment_v2/test_set.csv"
    train_data = training(train_url)
    test_y, y = test_result(test_url, train_data)
    cm.print_data(test_y, y)


main()
