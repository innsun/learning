# coding=utf-8
import struct
import numpy as np
import pandas as pd

def load_mnist(url, is_image=True):
    """ 加载mnist数据集 """
    with open(url, "rb") as f:
        buffers = f.read()
        type = ">IIII" if is_image else ">II"  # 读取文件头：4个或2个unsinged int32
        head = struct.unpack_from(type, buffers, 0)
        offset = struct.calcsize(type)
        num = head[1]
        fetures = head[2] * head[3] if is_image else 1
        bits = num * fetures
        bits_string = '>' + str(bits) + 'B'  # 读取文件内容：bits个unsigned byte

        data = struct.unpack_from(bits_string, buffers, offset)
        if is_image:
            data = map(lambda i: 1 if i > 100 else 0, data)  # 图片像素值处理，可根据自己需要调整
        data = np.reshape(data, [num, fetures])
        print("load minist finished：" + url)
        return data


def get_classify(x, y, num):
    """ 获取数字分类 """
    new_x, new_y = [], []
    for i, val in enumerate(y):
        if val < num:
            new_x.append(x[i])
            new_y.append(val)
    temp = np.zeros((len(new_y), num))
    for i, val in enumerate(new_y):
        temp[i][val] = 1  # 将0~9的数字映射到长度为10的数组上
    new_y = temp
    return np.array(new_x), new_y

def load_data(url, ignore_row):
    """ 加载数据。 """
    data = pd.read_csv(url)
    x = data.iloc[ignore_row:, 1:17].values
    y = data.iloc[ignore_row:, 17:].values
    # x = np.hstack((x, x * x))
    return x, y