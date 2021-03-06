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
from sklearn.cluster import DBSCAN

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

    def dbscan(self,data,eps,m,dis='l2'):
        data_nb = len(data)
        if dis == 'l1':
            dis_f = self.dis_l1
        elif dis == 'l2':
            dis_f = self.dis_l2
        else :
            print('Error please select correct reg')
            return
        #距离矩阵
        dis_matrix = np.zeros(shape=[data_nb,data_nb])
        for i in range(data_nb):
            dis_matrix[i] = dis_f(data[i],data)

        #已经遍历的点集合
        reached_list = []

        #计算可达点
        def getReachable(x_index):
            return [i for i,x in enumerate(dis_matrix[x_index]) if x<=eps and (i not in reached_list)]

        #核心点集合
        core_list = [i for i in range(data_nb) if len(getReachable(i))>=m]

        if len(core_list) <= 0:
            return
        #簇
        cluster = []
        while core_list!=[]:
            temp = []
            qeuen = []#队列
            core = random.sample(core_list,1)#随机选择核心点
            qeuen.extend(core)
            while len(qeuen)>0:
                p = qeuen.pop(0)#弹出队列中第一个元素
                if p in core_list:
                    core_list.remove(p)
                p_reach = getReachable(p)
                qeuen.extend(p_reach)#将核心点的可达点加入队列中
                reached_list.extend(p_reach)
#                 np.delete(dis_matrix,p_reach,axis=1)
                temp.extend(p_reach)
            cluster.append(temp)#一个核心点的所有可达点形成一个簇

        #判断是否有噪声点
        if len(set(range(data_nb))-set(reached_list))>0:
            noise = list(set(range(data_nb))-set(reached_list))
            cluster.append(noise)

        res = [0]*data_nb
        for i,v in enumerate(cluster):
            for c in v:
                res[c]=i

        return res

if __name__ == "__main__":
    N = 400
    centers = 4
    data, y = ds.make_blobs(N, n_features=2, centers=centers, random_state=2)

    #sklearn Kmeans
    cls = KMeans(n_clusters=4, init='k-means++')
    y_khat = cls.fit_predict(data)

    #sklearn DBSCAN
    model = DBSCAN(eps=0.8, min_samples=5)
    model.fit(data)
    y_dhat = model.labels_


    #My clustering
    mcls = MyClustering()
    y_mkhat = mcls.k_means(data,4,t=0.0005,dis='l2')
    y_mdhat = mcls.dbscan(data,0.8,5)


    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    cm = matplotlib.colors.ListedColormap(list('rgbm'))

    plt.figure(figsize=(8, 10), facecolor='w')
    plt.subplot(511)
    plt.title(u'原始数据')
    plt.scatter(data[:, 0], data[:, 1], c=y, s=30, cmap=cm, edgecolors='none')


    def expand(a, b):
        d = (b - a) * 0.1
        return a-d, b+d

    x1_min, x2_min = np.min(data, axis=0)
    x1_max, x2_max = np.max(data, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(512)
    plt.title(u'KMeans++聚类')
    plt.scatter(data[:, 0], data[:, 1], c=y_khat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(513)
    plt.title(u'DBSCAN聚类')
    plt.scatter(data[:, 0], data[:, 1], c=y_dhat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(514)
    plt.title(u'MyKMeans 聚类')
    plt.scatter(data[:, 0], data[:, 1], c=y_mkhat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(515)
    plt.title(u'MyDBSCAN 聚类')
    plt.scatter(data[:, 0], data[:, 1], c=y_mdhat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.suptitle(u'聚类算法对比结果', fontsize=18)
    plt.show()