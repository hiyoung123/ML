#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author: hiyoung 
@file: Regression.py 
@time: 2018/10/09
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

class MyLinearRegression(object):
    def __init__(self):
        self.theta = [0]
        self.loss = []

    def fit(self,x_data,y_data,alpha=0.01,iters=1):
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        feat_n = x_data.shape[1]#特征数
        x_data = np.insert(x_data,feat_n-1,1,axis=1)#对数据集x插入一列1
        self.theta = self.theta*(feat_n+1)#对Θ加入b

        #随机梯度下降
        for i in range(0,iters):
            for x,y in zip(x_data,y_data):
                pre = self.__objective(x)
                error = y-pre
                self.theta = self.theta+np.multiply(alpha*error,x)
                self.loss.append(self.coast(x[:-1].reshape(-1, feat_n), y))
        print('fit done!')

    def __objective(self,x):
        return np.dot(x,self.theta)

    def predict(self,x):
        x = np.array(x)
        x = np.insert(x,x.shape[1]-1,1,axis=1)
        return self.__objective(x)

    def coast(self,x,y):
        return (1/2.0)*sum(np.power(self.predict(x)-y,2))

    def get_weight(self):
        return self.theta[:-1]

    def get_loss(self):
        return self.loss

    def get_interpcet(self):
        return self.theta[-1]

if __name__ == '__main__':
    # 读数据
    data_file = "../Advertising.csv"
    data = pd.read_csv(open(data_file))

    x = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    lr = LinearRegression()
    mlr = MyLinearRegression()
    ridge = Ridge()
    lasso = Lasso()
    lr.fit(x_train, y_train)
    mlr.fit(x_train, y_train, alpha=0.000009, iters=20)
    ridge.fit(x_train, y_train)
    lasso.fit(x_train, y_train)
    #获取各个模型参数
    print('lr weights:', lr.coef_, ' intercept:', lr.intercept_)
    print('mlr weights:', mlr.get_weight(), ' intercept:', mlr.get_interpcet())
    print('lasso weights:', lasso.coef_, ' intercept:', lasso.intercept_)
    print('ridge weights:', ridge.coef_, ' intercept:', ridge.intercept_)

    lr_pre_test = lr.predict(x_test)
    mlr_pre_test = mlr.predict(x_test)
    lasso_pre_test = lasso.predict(x_test)
    ridge_pre_test = ridge.predict(x_test)
    #计算平方误差
    print('mlr error:', np.average((mlr_pre_test - np.array(y_test)) ** 2))
    print('lr error:', np.average((lr_pre_test - np.array(y_test)) ** 2))
    print('lasso error:', np.average((lasso_pre_test - np.array(y_test)) ** 2))
    print('ridge error:', np.average((ridge_pre_test - np.array(y_test)) ** 2))
    #对比图像
    plt.figure(facecolor='w', figsize=(12, 6))
    t = np.arange(len(x_test))
    plt.plot(t, lr_pre_test, 'g-', label=u'LR预测数据')
    plt.plot(t, y_test, 'r-', label=u'真实数据')
    plt.plot(t, mlr_pre_test, 'b-', label=u'MLR预测数据')
    plt.plot(t, lasso_pre_test, 'k-', label=u'lasso预测数据')
    plt.plot(t, ridge_pre_test, 'm-', label=u'ridge预测数据')
    plt.legend(loc='lower right')
    plt.title(u'线性回归')
    plt.grid()
    plt.show()