#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/9 9:19
# @Author  : Jack Zhao
# @Site    : 
# @File    : result.py
# @Software: PyCharm

# #Desc:
from os.path import dirname, join as pjoin
import scipy.io as sio

mat_name = "/data/CrossRec/log/douban_mb/logt/CrossRec_lenovo_shop_lenovo_community_KSize_64_Result.mat"
mat_contents = sio.loadmat(mat_name)
print(mat_contents.keys())
print(mat_contents['allResults_B'])
