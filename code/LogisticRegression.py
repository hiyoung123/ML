#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author: hiyoung 
@file: LogisticRegression.py
@time: 2018/10/28
"""
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
class MyLogistic(object):
    def __init__(self):
        self.theta = [0]

    def fit(self,x_data,y_data,iters=500,alpha=0.001):
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        feat_n = x_data.shape[1]  # 特征数
        x_data = np.insert(x_data, feat_n - 1, 1, axis=1)  # 对数据集x插入一列1
        self.theta = self.theta * (feat_n + 1)  # 对Θ加入b

        # 随机梯度下降
        for i in range(0, iters):
            for x, y in zip(x_data, y_data):
                pre = self.__sigmoid(self.__objective(x))
                error = y - pre
                self.theta = self.theta + np.multiply(alpha * error, x)
        print('fit done!')

    def __sigmoid(selfm,z):
        return 1.0 / (1 + np.exp(-z))

    def __objective(self,x):
        return np.dot(x,self.theta)

    def predict_prob(self,x):
        x = np.array(x)
        x = np.insert(x,x.shape[1]-1,1,axis=1)
        return self.__sigmoid(self.__objective(x))

    def predict(self,x):
        prob = self.predict_prob(x)
        res = []
        for i in prob:
            if i >= 0.5:
                res.append(1)
            else:
                res.append(0)
        return np.array(res)

    def get_weight(self):
        return self.theta[:-1]

    def get_interpcet(self):
        return self.theta[-1]

if __name__ == "__main__":
    path = 'data/iris.data'
    data = pd.read_csv(path, header=None)
    data[4] = pd.Categorical(data[4]).codes
    data = data.as_matrix()
    data = np.array([i for i in data if i[-1] != 2]) #只是用两个类别
    x = data[:,:-1]
    y = data[:,-1]
    mlr = MyLogistic()
    lr = LogisticRegression()
    mlr.fit(x,y)
    lr.fit(x, y)
    m_y_hat = mlr.predict(x)
    y_hat = lr.predict(x)

    print('原始值:{}'.format(y))
    print('my lr:{}'.format(m_y_hat))
    print('sklearn lr:{}'.format(y_hat))
    print('my lr 准确度：{}'.format(100 * np.mean( m_y_hat == y)))
    print('sklearn lr 准确度：{}'.format(100 * np.mean(y_hat == y)))




