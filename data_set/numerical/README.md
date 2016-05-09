# 数据说明

1、源数据来自num_v1，大家直接使用train_set.csv和test_set.csv即可
2、训练集500个样本，测试集100个样本
3、样本为像素为18*18，0~9的数字（按像素点存储1或0）
4、数据读取方式有调整（前10列为y值）
def load_data(url):
    data = pd.read_csv(url)
    y = data.iloc[:, 0:10]
    x = data.iloc[:, 10:]
    return x.values, y.values