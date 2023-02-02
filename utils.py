#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/24 20:30
# @Author  : Jack Zhao
# @Site    : 
# @File    : utils.py
# @Software: PyCharm

# #Desc: 一些工具函数
import math
import torch
import time

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from config import opt
from torch.autograd import grad
from functools import wraps


def mask_edge(testA, testB, src, tgt, index_gap):
    """求差集，用于图"""
    pairs_A = np.array(testA)[:, :2].astype('int')  # 前两列
    pairs_B = np.array(testB)[:, :2].astype('int')  # 前两列
    pairs_B[:, 1] = index_gap + pairs_B[:, 1]
    pairs_A = pd.DataFrame({'u': pairs_A[:, 0], 'i': pairs_A[:,1],'r':0})
    pairs_B = pd.DataFrame({'u': pairs_B[:, 0], 'i': pairs_B[:, 1],'r':0})

    src = set_diff_df(src,pairs_A)
    tgt = set_diff_df(tgt,pairs_B)
    return src, tgt

def mask_data(train_u,train_i,train_r,overlap_mask):
    data = pd.DataFrame()
    data['users'], data['items'], data['ratings'] = train_u,train_i,train_r
    data = data.loc[~data['users'].isin(overlap_mask)]
    data_overlap = data.loc[data['users'].isin(overlap_mask)] # 用于图过滤
    train_len = data.shape[0]
    return data['users'].values, data['items'].values, data['ratings'].values,train_len,data_overlap


# def mask_edge_origin(testA, testB, src, tgt, item_src_map, item_tgt_map):
#     """求差集"""
#     pairs_A = np.array(testA)[:, :2].astype('int')  # 前两列
#     pairs_B = np.array(testB)[:, :2].astype('int')  # 前两列
#     pairs_A[:, 1] = np.apply_along_axis(lambda x: int(item_src_map[str(x[1])]), axis=1, arr=pairs_A)  # item映射
#     pairs_B[:, 1] = np.apply_along_axis(lambda x: int(item_tgt_map[str(x[1])]), axis=1, arr=pairs_B)  # item映射
#     # src_pairs = src[['u', 'i']].values
#     # tgt_pairs = tgt[['u', 'i']].values
#     pairs_A = pd.DataFrame({'u': pairs_A[:, 0], 'i': pairs_A[:,1],'r':0})
#     pairs_B = pd.DataFrame({'u': pairs_B[:, 0], 'i': pairs_B[:, 1],'r':0})
#
#     src = set_diff_df(src,pairs_A)
#     tgt = set_diff_df(tgt,pairs_B)
#     return src,tgt

def set_diff(A,B):
    """numpy两个二维数组差集"""
    res = np.array(list(set(map(tuple, A)) - set(map(tuple, B))))
    return res

def set_diff_df(A,B):
    """pandas两个差集,A是大的"""
    A = pd.concat((A,B),axis=0,ignore_index=True)
    A = pd.concat((A,B),axis=0,ignore_index=True)
    A = A.drop_duplicates(subset=['u','i'],keep=False).reset_index(drop=True)
    return A

def emb_permutation(x,i_feature):
    """做embedding的置换"""




def create_feed_dict(u, i, dataset, r=None, drop=None,device=None):
        return {
            "user_A": torch.from_numpy(u).long().to(device),
            "item_A": torch.from_numpy(i).long().to(device),
            "rate_A": torch.from_numpy(r).to(device),
            "drop_A": drop,
            "user_B": torch.from_numpy(u).long().to(device),  
            "item_B": torch.from_numpy(i).long().to(device), # 这里最好传入
            "rate_B": torch.from_numpy(r).to(device),
            "drop_B": drop
        }



def add_embedding_matrix(dataSet_A,dataSet_B):
    """创建MF emb，可以不用图"""
    user_item_embedding_A = torch.from_numpy(dataSet_A.getEmbedding())  # u*i rating矩阵
    item_user_embedding_A = user_item_embedding_A.T
    user_item_embedding_B = torch.from_numpy(dataSet_B.getEmbedding())
    item_user_embedding_B = user_item_embedding_B.T

    return user_item_embedding_A,item_user_embedding_A,user_item_embedding_B,item_user_embedding_B


def getHitRatio(ranklist, targetItem):
    for item in ranklist:
        if item == targetItem:
            return 1
    return 0


def getNDCG(ranklist, targetItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == targetItem:
            return math.log(2) / math.log(i + 2)
    return 0


def l2_loss(tensor):
    loss = torch.sum(tensor.square())
    return loss

def l2_loss_mean(tensor):
    loss = torch.mean(tensor.square())
    return loss

def criterion(pred, label):
    loss = label * torch.log(pred) + (1 - label) * torch.log(1 - pred)
    return loss

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def print_info(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration_time = end_time - start_time
        print("execute time running %s: %s seconds" % (func.__name__, duration_time))
        return result

    return wrapper



def activation_layer(act_name, hidden_size=None, dice_dim=2):
    """Construct activation layers

    Args:
        act_name: str or nn.Module, name of activation function
        hidden_size: int, used for Dice activation
        dice_dim: int, used for Dice activation
    Return:
        act_layer: activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'linear':
            act_layer = Identity()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'leakyrelu':
            act_layer = nn.LeakyReLU(inplace=True)
        elif act_name.lower() == 'dice':
            assert dice_dim
            act_layer = Dice(hidden_size, dice_dim)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer


class Identity(nn.Module):
    """返回原来"""
    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


class Dice(nn.Module):
    """The Data Adaptive Activation Function in DIN,which can be viewed as a generalization of PReLu and can adaptively adjust the rectified point according to distribution of input data.

    Input shape:
        - 2 dims: [batch_size, embedding_size(features)]
        - 3 dims: [batch_size, num_features, embedding_size(features)]

    Output shape:
        - Same shape as input.

    References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
        - https://github.com/zhougr1993/DeepInterestNetwork, https://github.com/fanoping/DIN-pytorch
    """

    def __init__(self, emb_size, dim=2, epsilon=1e-8, device='cpu'):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3

        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

        # wrap alpha in nn.Parameter to make it trainable
        if self.dim == 2:
            self.alpha = nn.Parameter(torch.zeros((emb_size,)).to(device))
        else:
            self.alpha = nn.Parameter(torch.zeros((emb_size, 1)).to(device))

    def forward(self, x):
        assert x.dim() == self.dim
        if self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        else:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)
        return out



@torch.no_grad()
def sinkhorn(out):
    """保证q的约束,使得信息传输过程中熵最小"""
    Q = torch.exp(out / opt.EPSILON).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * (-1) # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(opt.SI):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

def reg(tensor):
    return torch.sqrt(torch.sum(torch.square(tensor), axis=1) + 1e-8)


def gradient_discrepancy_loss_margin(info, regRate_A,regRate_B, netC1, netC2,loss_type):
    """没有类别"""
    loss_1 = 0
    loss_2 = 0
    # gm loss
    gm_loss = 0
    grad_cossim = []
    user_out_A_, item_out_A, user_out_B_, item_out_B, norm_item_output_A, norm_item_output_B = info

    if loss_type =="A":
        user_out_A,user_out_B = netC1(user_out_A_), netC1(user_out_B_)
        norm_user_output_A = reg(user_out_A)
        norm_user_output_B = reg(user_out_B)
        y_1 = torch.sum(
                torch.multiply(user_out_A, item_out_A), axis=1,
                keepdims=False) / norm_item_output_A * norm_user_output_A
        y_1 = torch.clamp(y_1,1e-6)
        y_2 = torch.sum(
                torch.multiply(user_out_B, item_out_A), axis=1,
                keepdims=False) / norm_item_output_A * norm_user_output_B
        y_2 = torch.clamp(y_2,1e-6)
        loss_1, loss_2 = -torch.sum(criterion(y_1, regRate_A)), -torch.sum(
            criterion(y_2, regRate_A))  # 这是score.
        # netE+C1
        for n, p in netC1.named_parameters():
            real_grad = grad([loss_1],
                             [p],
                             create_graph=True,
                             only_inputs=True,
                             allow_unused=False)[0]
            fake_grad = grad([loss_2],
                             [p],
                             create_graph=True,
                             only_inputs=True,
                             allow_unused=False)[0]

            if len(p.shape) > 1:
                _cossim = F.cosine_similarity(fake_grad, real_grad, dim=1).mean()
            else:
                _cossim = F.cosine_similarity(fake_grad, real_grad, dim=0)
            grad_cossim.append(_cossim)

        grad_cossim1 = torch.stack(grad_cossim)
        gm_loss = (1.0 - grad_cossim1).mean()
    else:
        user_out_A,user_out_B = netC2(user_out_A_), netC2(user_out_B_)
        norm_user_output_A = reg(user_out_A)
        norm_user_output_B = reg(user_out_B)
        y_1 = torch.sum(
            torch.multiply(user_out_B, item_out_B), axis=1,
            keepdims=False) / norm_item_output_B * norm_user_output_B
        y_1 = torch.clamp(y_1,1e-6)
        y_2 = torch.sum(
            torch.multiply(user_out_A, item_out_B), axis=1,
            keepdims=False) / norm_item_output_B * norm_user_output_A
        y_2 = torch.clamp(y_2,1e-6)
        loss_1, loss_2 = -torch.sum(criterion(y_1, regRate_B)), -torch.sum(
            criterion(y_2, regRate_B))  # 这是score.

        # netE+C2
        for n, p in netC2.named_parameters():
            real_grad = grad([loss_1],
                             [p],
                             create_graph=True,
                             only_inputs=True)[0]
            fake_grad = grad([loss_2],
                             [p],
                             create_graph=True,
                             only_inputs=True)[0]

            if len(p.shape) > 1:
                _cossim = F.cosine_similarity(fake_grad, real_grad, dim=1).mean()
            else:
                _cossim = F.cosine_similarity(fake_grad, real_grad, dim=0)
            grad_cossim.append(_cossim)

        grad_cossim2 = torch.stack(grad_cossim)
        gm_loss = (1.0 - grad_cossim2).mean()

    return gm_loss


if __name__ == '__main__':
    # test = torch.randn(3,3)
    # out = sinkhorn(test)
    # print(test,out)
    # A = np.array([[1,2,3],[4,5,6]])
    # B = np.array([[1,2,3]])
    # set_diff(A,B)
    #
    A = pd.DataFrame({'u':[0,1,2],'i':[0,2,3],'r':[0,2,1]})
    B = pd.DataFrame({'u':[1,2,3],'i':[2,3,4],'r':[1,2,3]})
    set_diff_df(A,B)
