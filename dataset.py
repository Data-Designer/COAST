#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/24 20:56
# @Author  : Jack Zhao
# @Site    : 
# @File    : dataset.py
# @Software: PyCharm

# #Desc: Customized Dataset
import numpy as np
import pandas as pd
from copy import deepcopy

class Dataset():
    def __init__(self, fileName):
        self.data, self.shape = self.getData(fileName)
        self.train, self.test = self.getTrainTest()
        self.trainDict = self.getTrainDict()

    def getData_1(self, fileName):
        """获取数据集dataset,原始"""
        print("Loading %s data set..." % (fileName))
        data = []
        filePath = fileName + '/ratings.dat'
        u, i, maxr = 0, 0, 0.0
        with open(filePath, 'r') as f:
            for line in f:
                if line:
                    lines = line.split("\t")
                    user,movie,score = int(lines[0]),int(lines[1]),float(lines[2])
                    data.append((user, movie, score, 0))
                    if user > u: # 存储的是最大的index
                        u = user
                    if movie > i:
                        i = movie
                    if score > maxr:
                        maxr = score

        self.maxRate = maxr
        print("Loading Success!\n"
              "Data Info:\n"
              "\tUser Num: {}\n"
              "\tItem Num: {}\n"
              "\tData Size: {}".format(u, i, len(data)))
        return data, [u, i]


    def getData(self, fileName):
        """获取数据集dataset，改进"""
        print("Loading %s data set..." % (fileName))
        data = []
        filePath = fileName + '/ratings_p.csv'
        u, i, maxr = 0, 0, 0.0
        file = pd.read_csv(filePath,index_col=0,header=0)
        file.columns = ["user_id","item_id","score"]
        file["user_id"] = file["user_id"].astype(int)
        file["item_id"] = file["item_id"].astype(int)
        file["score"] = file["score"].astype(float)
        u,i,maxr = file["user_id"].max(),file["item_id"].max(),file["score"].max()
        data = list(zip(file["user_id"].values.tolist(),file["item_id"].values.tolist(),file["score"].values.tolist(),[0]*file.shape[0]))

        self.maxRate = maxr
        print("Loading Success!\n"
              "Data Info:\n"
              "\tUser Num: {}\n"
              "\tItem Num: {}\n"
              "\tData Size: {}".format(u, i, len(data)))
        return data, [u, i]



    def getTrainTest(self):
        """切分数据集, 注意元数据是从1开始编码的"""
        data = self.data
        data = sorted(data, key=lambda x: (x[0], x[3])) # user score从小到大排序
        train, test = [], []
        for i in range(len(data) - 1):
            user = data[i][0] - 1 # start from zero (lenovo)
            item = data[i][1] - 1
            rate = data[i][2]
            if data[i][0] != data[i + 1][0]:
                test.append((user, item, rate))
            else:
                train.append((user, item, rate)) # 留一法

        test.append((data[-1][0] - 1, data[-1][1] - 1, data[-1][2]))
        return train, test

    def getTrainDict(self):
        """train中的交互项"""
        dataDict = {}
        for i in self.train:
            dataDict[(i[0], i[1])] = i[2]
        return dataDict

    def getEmbedding(self):
        """返回np array,用于lookup embedding"""
        train_matrix = np.zeros([self.shape[0], self.shape[1]], dtype=np.float32)
        for i in self.train:
            user = i[0]
            movie = i[1]
            rating = i[2]
            train_matrix[user][movie] = rating # 交互矩阵，shape==[u，i]
        return np.array(train_matrix)

    def getInstance(self,data,negNum):
        """以数组形式返回单一的列，首位是正例，训练时的负采样策略"""
        user,item,rate = [],[],[]
        for i in data:
            user.append(i[0])
            item.append(i[1])
            rate.append(i[2])
            for t in range(negNum):
                j = np.random.randint(self.shape[1])
                while (i[0], j) in self.trainDict: # 这里随机采样万一重复咋办？？
                    j = np.random.randint(self.shape[1])
                user.append(i[0])
                item.append(j)
                rate.append(0.0)
        return np.array(user), np.array(item), np.array(rate)

    def getTestNeg(self,testData, negNum):
        """Test Negative sampling method"""
        user,item = [],[]
        for s in testData: # each user
            tmp_user,tmp_item = [], []
            u,i = s[0],s[1]
            tmp_user.append(u) # 第一位都是正样本
            tmp_item.append(i)
            neglist = set()
            neglist.add(i)
            for t in range(negNum):
                j = np.random.randint(self.shape[1]) # 
                while (u, j) in self.trainDict or j in neglist: # 确保不重复
                    j = np.random.randint(self.shape[1])
                neglist.add(j)
                tmp_user.append(u)
                tmp_item.append(j)
            user.append(tmp_user)
            item.append(tmp_item)
        return [np.array(user), np.array(item)] # [[1,1,1...],[2,2,2...]] user*100

