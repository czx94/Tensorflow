# -*- coding: utf-8 -*-
import pickle as p
import numpy as np
import os
from numpy.random import randint


def load_CIFAR_batch(filename):
    """ 载入cifar数据集的一个batch """
    with open(filename, 'rb') as f:
        datadict = p.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ 载入cifar全部数据 """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X) #将所有batch整合起来
        ys.append(Y)
        Xtr = np.concatenate(xs) #使变成行向量,最终Xtr的尺寸为(50000,32,32,3)
        Ytr = np.concatenate(ys)
        del X, Y
    Xte, label = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    test_label = np.zeros((len(label), 10))
    for i in range(len(label)):
        test_label[i][label[i]] = 1

    return Xtr, Ytr, Xte, test_label

def data_generator(x, y, batch_size):
    count = 0
    loopcount = len(y) // batch_size
    while 1:
        j = randint(0, loopcount)
        train_data = x[j*batch_size: (j+1)*batch_size]
        label = y[j*batch_size: (j+1)*batch_size]
        train_label = np.zeros((batch_size, 10))
        for i in range(batch_size):
            train_label[i][label[i]] = 1
        yield train_data, train_label
        count += batch_size

if __name__ == '__main__':
    # 载入CIFAR-10数据集
    cifar10_dir = './cifar-10-batches-py/'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    data_gen = data_generator(X_train,y_train,32)
    for i in range(30):
        a, b = next(data_gen)



# # 看看数据集中的一些样本：每个类别展示一些
# print('Training data shape: ', X_train.shape)
# print('Training labels shape: ', y_train.shape)
# print('Test data shape: ', X_test.shape)
# print('Test labels shape: ', y_test.shape)