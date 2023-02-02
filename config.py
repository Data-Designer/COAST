#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/24 20:30
# @Author  : Jack Zhao
# @Site    : 
# @File    : config.py
# @Software: PyCharm

# #Desc: Cross Rec配置函数

import warnings

class TrainConfig():
    """注意lenovo这里要去掉跨域u-i交互"""
    ABA = False # 是否进行Discussion，例如overlap
    Ratio = 0.25 # mask掉的
    GCN_LAYER = 2 # 2交互过多没啥用
    GPU_USED = True
    MEMENTUM = 0.1 # 动量更新
    LR = 5e-4 # mb 5e-4, mm 2e-4,lenovo 7e-4
    GAMMA = 0.0003
    DECAY = 0.75
    WEIGHT_DECAY = 5e-4
    BATCH_SIZE = 4096 # 4096
    SHUFFLE = True
    EPOCHS = 120 #
    NEARK = 8 # 无用
    NEGNUM = 7 # 7,
    TOPK = 10
    KSIZE = 64 # 64
    LAMBDA = 1e-3 # 矩阵正则化 # 1e-3
    LAMBDA2 = 1e-2 # swav正则化 # 1e-2
    LAMBDA3 = 1
    START = -1
    EPSILON = 0.05 # regularization parameter for Sinkhorn-Knopp algorithm
    SI = 3 # sinkhorn_iterations
    PROTONUM = 256 # 原型数量 256, lenovo 64
    TEMPERATURE = 0.1 # temperature parameter in training loss
    FREZZE = 0 # swav正则化 200
    DATAA = "/data/CrossRec/data/douban_movie"
    DATAB = "/data/CrossRec/data/douban_book"
    LOGFILE = "/data/CrossRec/log/douban_mb/logt/"
    TLOGFILE = "/data/CrossRec/log/douban_mb/logt/run"
    WEIGHTS = "/data/CrossRec/log/douban_mb/logt/checkpoint/"
    CASEFILE = "/data/CrossRec/log/douban_mb/logt/case.csv"
    SEED = 1
    LOG_STEP_FREQ = 500 # Epoch中间进程没有必要打印

#
# class TrainConfig_batch():
#     """这里是IMage CLEF的配置项,注意修改生成器features.parameters为parameters"""
#     ABA = False # 是否进行Discussion，例如overlap
#     GCN_LAYER = 2
#     GPU_USED = True
#     MEMENTUM = 0.1 # 动量更新
#     LR = 2e-4 # mb 5e-4
#     GAMMA = 0.0003
#     DECAY = 0.75
#     WEIGHT_DECAY = 5e-4
#     BATCH_SIZE = 4096 # 4096
#     SHUFFLE = True
#     EPOCHS = 50 # 第一个epoch使用source进行初始化
#     NEARK = 5
#     NEGNUM = 7
#     TOPK = 10
#     KSIZE = 64
#     LAMBDA = 0.001 # 矩阵正则化
#     LAMBDA2 = 0.01 # swav正则化
#     START = -1
#     EPSILON = 0.05 # regularization parameter for Sinkhorn-Knopp algorithm
#     SI = 3 # sinkhorn_iterations
#     PROTONUM = 256 # 原型数量
#     TEMPERATURE = 0.1 # temperature parameter in training loss
#     FREZZE = 0 # swav正则化 start=200
#     DATAA = "/data/CrossRec/data/douban_movie"
#     DATAB = "/data/CrossRec/data/douban_music"
#     LOGFILE = "/data/CrossRec/log/douban_mb/logt/"
#     TLOGFILE = "/data/CrossRec/log/douban_mb/logt/run"
#     WEIGHTS = "/data/CrossRec/log/douban_mb/logt/checkpoint/"
#     CASEFILE = "/data/CrossRec/log/douban_mb/logt/case.csv"
#     SEED = 1
#     LOG_STEP_FREQ = 500 # Epoch中间进程没有必要打印

def parse(self, kwargs):
    '''
    根据字典kwargs 更新 config参数，可以自行设定一些参数
    '''
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


TrainConfig.parse = parse # 猴子补丁
opt = TrainConfig() # 包含配置项的实例