#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author: hiyoung 
@file: Clustering.py 
@time: 2018/10/19
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import KMeans

class MyClustering(object):
    def __init__(self):
        pass
    #L1 距离
    def dis_l1(self,x,y):
        return np.sum(np.abs(x-y),axis=1)
    #L2 距离
    def dis_l2(self,x,y):
        return np.sqrt(np.sum(np.power(x-y,2),axis=1))


    def k_means(self,data,n_clusters,t=0.1,dis='l2'):
        feat_nb = len(data[0])
        data_nb = len(data)

        if dis == 'l1':
            dis_f = self.dis_l1
        elif dis == 'l2':
            dis_f = self.dis_l2
        else :
            print('Error please select correct reg')
            return

        centers = data[random.sample(range(data_nb),n_clusters)]#随机选取初始中心点

        done = False
        cluster = {k:[] for k in range(n_clusters)}
        while not done:
            for i in range(data_nb):
                distance = dis_f(data[i],centers)
                index = np.argmin(distance)
                cluster[index].append(i)

            done = True
            for i in range(n_clusters):
                new_center = np.mean(data[cluster[i]],axis=0)
                if dis_f(new_center.reshape(-1,feat_nb),centers[i].reshape(-1,feat_nb)) > t:
                    done = False #只要有一个簇类没有达到要求阈值，就继续迭代更新
                centers[i] = new_center #更新中心点

        #res 返回每个样本的簇类
        res = [0]*data_nb
        for i,v in cluster.items():
            for c in v:
                res[c]=i

        return res

if __name__ == "__main__":
    N = 400
    centers = 4
    data, y = ds.make_blobs(N, n_features=2, centers=centers, random_state=2)

    cls = KMeans(n_clusters=4, init='k-means++')
    y_hat = cls.fit_predict(data)

    mcls = MyClustering()
    y_mhat = mcls.k_means(data,n_clusters=4,t=0.0005, dis='l2')

    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    cm = matplotlib.colors.ListedColormap(list('rgbm'))

    plt.figure(figsize=(11, 10), facecolor='w')
    plt.subplot(311)
    plt.title(u'原始数据')
    plt.scatter(data[:, 0], data[:, 1], c=y, s=30, cmap=cm, edgecolors='none')
    x1_min, x2_min = np.min(data, axis=0)
    x1_max, x2_max = np.max(data, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(312)
    plt.title(u'KMeans++聚类')
    plt.scatter(data[:, 0], data[:, 1], c=y_hat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(313)
    plt.title(u'MyKMeans 聚类')
    plt.scatter(data[:, 0], data[:, 1], c=y_mhat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.suptitle(u'数据分布对KMeans聚类的影响', fontsize=18)
    plt.show()