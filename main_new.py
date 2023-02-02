#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/24 20:16
# @Author  : Jack Zhao
# @Site    : 
# @File    : main.py
# @Software: PyCharm

# #Desc: 这里的index重新修正,NGCF修正，data_u_info修正 latest
import os
package_name = 'gensim'
os.system(f'pip install -i https://pypi.tuna.tsinghua.edu.cn/simple {package_name}==3.8.3')

import torch
import numpy as np
import sys
import heapq
import operator
import dgl

import pandas as pd
import scipy.io as scio
import torch.nn as nn


from config import opt
from dataset import Dataset
from gensim.models.doc2vec import Doc2Vec
from model.models import Contra_Two_Tower,create_graph,neighbor_table,graph_hop_two,GraphNet,read_graph,GraphNet_NGCF,dense_to_sparse,aggravate_domain,graph_hop_sep
from utils import create_feed_dict,getNDCG,getHitRatio,add_embedding_matrix,criterion,mask_edge,mask_data,gradient_discrepancy_loss_margin

# torch.manual_seed(opt.SEED)
# if opt.GPU_USED:
#     torch.cuda.manual_seed(opt.SEED)



def train_step(iter, model,feed_dict,inter_embedding_dict,graph_data,origin_emb, neighbour_table,data_info,pretrain,loss_type):
    """模型训练"""
    model.train()
    # 梯度清零
    model.optimizer.zero_grad()
    # 数据分解,获取embedding
    user_A,item_A,rate_A,drop_A,user_B,item_B,rate_B,drop_B = feed_dict['user_A'],feed_dict['item_A'],feed_dict['rate_A'],feed_dict['drop_A'], \
                                                              feed_dict['user_B'],feed_dict['item_B'],feed_dict['rate_B'],feed_dict['drop_B']

    # 传统MF
    # user_input_A,item_input_A,user_input_B,item_input_B = inter_embedding_dict[0](user_A),inter_embedding_dict[1](item_A),inter_embedding_dict[2](user_B),inter_embedding_dict[3](item_B)

    # 图表征emb
    # user_input_A, user_input_B = model_N2V_A(user_A),model_N2V_B(user_B)
    # item_input_A, item_input_B = model_N2V_A(item_A+data_info['shape_A'][0]), model_N2V_B(item_B+data_info['shape_B'][0]) # 因为图的ID是有序排列的。

    # 两阶段
    # user_input_A, user_input_B = torch.index_select(model_N2V_A, 0, user_A),\
    #                              torch.index_select(model_N2V_B, 0, user_B)
    # item_input_A, item_input_B = torch.index_select(model_N2V_A,0, item_A + data_info['shape_A'][0]),\
    #                              torch.index_select(model_N2V_B,0, item_B + data_info['shape_B'][0])  # 因为图的ID是有序排列的。

    # 端到端
    src_u_emb,tgt_u_emb,src_i_emb,tgt_i_emb = graph_data
    o_src_u_emb, o_tgt_u_emb, o_src_i_emb, o_tgt_i_emb = origin_emb
    neighbors_utable_src, neighbors_utable_tgt, neighbors_itable_src, neighbors_itable_tgt = neighbour_table # u*u
    user_input_A, user_input_B = src_u_emb(user_A), tgt_u_emb(user_B)
    user_input_A_o, user_input_B_o = o_src_u_emb(user_A), o_tgt_u_emb(user_B)
    user_input_A,user_input_B = torch.cat((user_input_A,user_input_A_o),dim=1),torch.cat((user_input_B,user_input_B_o),dim=1)


    if loss_type == 'A':
        item_input_A, item_input_B = src_i_emb(item_A), src_i_emb(item_B) 
        item_input_A_o, item_input_B_o = o_src_i_emb(item_A), o_src_i_emb(item_B) 
        item_input_A, item_input_B = torch.cat((item_input_A,item_input_A_o),dim=1),torch.cat((item_input_B,item_input_B_o),dim=1)
    else:
        item_input_A, item_input_B = tgt_i_emb(item_A), tgt_i_emb(item_B)
        item_input_A_o, item_input_B_o = o_tgt_i_emb(item_A), o_tgt_i_emb(item_B) 
        item_input_A, item_input_B = torch.cat((item_input_A,item_input_A_o),dim=1),torch.cat((item_input_B,item_input_B_o),dim=1)

    # origin_input = [user_input_A_o,user_input_B_o,item_input_A_o,item_input_B_o]


    regularizer_A, regularizer_B, user_swav_A, user_swav_B, y_A, y_B, info = model(user_input_A,user_input_B,item_input_A,item_input_B)

    regRate_A, regRate_B = rate_A / data_info["maxRate_A"], rate_B / data_info["maxRate_B"]

    # print(y_A.shape,regRate_A.shape)
    losses_A, losses_B = model.criterion(y_A, regRate_A), model.criterion(y_B, regRate_B) # 这是score.
    loss_A,loss_B = -torch.sum(losses_A), -torch.sum(losses_B)

    # print("loss_A:{},regularizer_A:{},user_swav_A:{}".format(loss_A.detach().item(), regularizer_A.detach().item(),
    #                                                          user_swav_A.detach().item()))
    gmn_loss = 0
    # 增加grad对齐
    net_1 = model.user_A_dnn
    net_2 = model.user_B_dnn
    gmn_loss = gradient_discrepancy_loss_margin(info,regRate_A,regRate_B, net_1, net_2,loss_type)
    loss_A,loss_B = loss_A + opt.LAMBDA * regularizer_A + opt.LAMBDA2 * user_swav_A + opt.LAMBDA3 * gmn_loss, loss_B + opt.LAMBDA * regularizer_B+opt.LAMBDA2 * user_swav_B + opt.LAMBDA3 * gmn_loss

    if pretrain == True: 
        loss_A, loss_B = user_swav_A, user_swav_B
    else:
        loss_A, loss_B = loss_A + opt.LAMBDA * regularizer_A + opt.LAMBDA2 * user_swav_A , loss_B + opt.LAMBDA * regularizer_B+opt.LAMBDA2 * user_swav_B

    if iter % opt.LOG_STEP_FREQ==0 and not pretrain:
        if loss_type == "A":
            print("loss_A:{},regularizer_A:{},user_swav_A:{},grad_A:{}".format(loss_A.detach().item(),regularizer_A.detach().item(),user_swav_A.detach().item(),gmn_loss))
        elif loss_type == "B":
            print("loss_B:{},regularizer_B:{},user_swav_B:{},grad_B:{}".format(loss_B.detach().item(),regularizer_B.detach().item(),user_swav_B.detach().item(),gmn_loss))
    elif iter % opt.LOG_STEP_FREQ==0 and pretrain:
        if loss_type == "A":
            print("user_swav_A:{}".format(user_swav_A.detach().item()))
        elif loss_type == "B":
            print("user_swav_B:{}".format(user_swav_B.detach().item()))

    # 反向传播求梯度
    if loss_type == "A":
        loss_A = loss_A * 0.4
        loss_A.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 5)  # 设置剪裁阈值为5
        loss = loss_A
        y = y_A
    else:
        loss_B = loss_B * 0.6
        loss_B.backward()
        loss = loss_B
        # torch.nn.utils.clip_grad_norm(model.parameters(), 5)  # 设置剪裁阈值为5
        y = y_B

    if iter < opt.FREZZE: 
        for name, p in model.named_parameters():
            if "prototypes" in name:
                p.grad = None

    model.optimizer.step()

    return loss.detach().item(), y



def evaluate(model,testNeg_A,testNeg_B,graph_data,origin_emb, neighbour_table,data_info,device):
    model.eval()
    with torch.no_grad():
        hr_A,NDCG_A = [],[]
        testUser_A, testItem_A = testNeg_A[0], testNeg_A[1]
        hr_B, NDCG_B = [], []
        testUser_B, testItem_B = testNeg_B[0], testNeg_B[1]

        for i in range(len(testUser_A)):
            target = testItem_A[i][0]
            feed_dict_A = create_feed_dict(testUser_A[i], testItem_A[i], 'A',np.array([]),device=device)

            user_A, item_A, rate_A, drop_A, user_B, item_B, rate_B, drop_B = feed_dict_A['user_A'], feed_dict_A['item_A'], \
                                                                             feed_dict_A['rate_A'], feed_dict_A['drop_A'], \
                                                                             feed_dict_A['user_B'], feed_dict_A['item_B'], \
                                                                             feed_dict_A['rate_B'], feed_dict_A['drop_B']

            # user_input_A, user_input_B = torch.index_select(model_N2V_A, 0, user_A),\
            #                              torch.index_select(model_N2V_B, 0,  user_B)
            # item_input_A, item_input_B = torch.index_select(model_N2V_A, 0,  item_A + data_info['shape_A'][0]), \
            #                              torch.index_select(model_N2V_B, 0, item_B + data_info['shape_B'][0])  # 因为图的ID是有序排列的。

            # 端到端
            src_u_emb, tgt_u_emb, src_i_emb, tgt_i_emb = graph_data
            o_src_u_emb, o_tgt_u_emb, o_src_i_emb, o_tgt_i_emb = origin_emb
            neighbors_utable_src, neighbors_utable_tgt, neighbors_itable_src, neighbors_itable_tgt = neighbour_table  # u*u
            user_input_A, user_input_B = src_u_emb(user_A), tgt_u_emb(user_B)
            user_input_A_o, user_input_B_o = o_src_u_emb(user_A), o_tgt_u_emb(user_B)
            user_input_A, user_input_B = torch.cat((user_input_A, user_input_A_o), dim=1), torch.cat(
                (user_input_B, user_input_B_o), dim=1)
            item_input_A, item_input_B = src_i_emb(item_A), src_i_emb(item_B)  
            item_input_A_o, item_input_B_o = o_src_i_emb(item_A), o_src_i_emb(item_B)  
            item_input_A, item_input_B = torch.cat((item_input_A, item_input_A_o), dim=1), torch.cat(
                (item_input_B, item_input_B_o), dim=1)


            regularizer_A, _, swav_A, _, predict_A, _,_ = model(user_input_A,user_input_B,item_input_A,item_input_B)

            item_score_dict = {}
            for j in range(len(testItem_A[i])): # 0+99=100
                item = testItem_A[i][j]
                item_score_dict[item] = predict_A[j]

            ranklist = heapq.nlargest(opt.TOPK,item_score_dict, key=item_score_dict.get)

            tmp_hr = getHitRatio(ranklist, target)
            tmp_NDCG = getNDCG(ranklist, target)
            hr_A.append(tmp_hr)
            NDCG_A.append(tmp_NDCG)

        for i in range(len(testUser_B)):
            target = testItem_B[i][0]
            feed_dict_B = create_feed_dict(testUser_B[i], testItem_B[i],'B',np.array([]),device=device) 
            user_A, item_A, rate_A, drop_A, user_B, item_B, rate_B, drop_B = feed_dict_B['user_A'], feed_dict_B['item_A'], \
                                                                             feed_dict_B['rate_A'], feed_dict_B['drop_A'], \
                                                                             feed_dict_B['user_B'], feed_dict_B['item_B'], \
                                                                             feed_dict_B['rate_B'], feed_dict_B['drop_B']

            # print(model_N2V_A.device,user_A.device)
            # user_input_A, user_input_B = torch.index_select(model_N2V_A, 0, user_A), \
            #                              torch.index_select(model_N2V_B, 0, user_B)
            # item_input_A, item_input_B = torch.index_select(model_N2V_A, 0, item_A + data_info['shape_A'][0]), \
            #                              torch.index_select(model_N2V_B, 0, item_B + data_info['shape_B'][0])  # 因为图的ID是有序排列的。

            # 端到端
            src_u_emb, tgt_u_emb, src_i_emb, tgt_i_emb = graph_data
            neighbors_utable_src, neighbors_utable_tgt, neighbors_itable_src, neighbors_itable_tgt = neighbour_table  # u*u
            user_input_A, user_input_B = src_u_emb(user_A), tgt_u_emb(user_B)
            user_input_A_o, user_input_B_o = o_src_u_emb(user_A), o_tgt_u_emb(user_B)
            user_input_A, user_input_B = torch.cat((user_input_A, user_input_A_o), dim=1), torch.cat(
                (user_input_B, user_input_B_o), dim=1)

            item_input_A, item_input_B = tgt_i_emb(item_A), tgt_i_emb(item_B)
            item_input_A_o, item_input_B_o = o_tgt_i_emb(item_A), o_tgt_i_emb(item_B) 
            item_input_A, item_input_B = torch.cat((item_input_A, item_input_A_o), dim=1), torch.cat(
                (item_input_B, item_input_B_o), dim=1)

            _, regularizer_B, _, swav_B, _, predict_B,_ = model(user_input_A, user_input_B, item_input_A,
                                                                               item_input_B)

            item_score_dict = {}
            for j in range(len(testItem_B[i])):
                item = testItem_B[i][j]
                item_score_dict[item] = predict_B[j]

            ranklist = heapq.nlargest(opt.TOPK,item_score_dict,key=item_score_dict.get)

            tmp_hr = getHitRatio(ranklist, target)
            tmp_NDCG = getNDCG(ranklist, target)
            hr_B.append(tmp_hr)
            NDCG_B.append(tmp_NDCG)

    return np.mean(hr_A), np.mean(NDCG_A), np.mean(hr_B), np.mean(NDCG_B)

def get_douban_graph_origin(dataset_A,dataset_B,testA, testB,data_uinfo,dataA_item_info,dataB_item_info,device):
    # 下面是图的数据，需要对齐
    dataA_file = opt.DATAA + '/ratings_p.csv'  # 注意转换
    dataB_file = opt.DATAB + '/ratings_p.csv'

    src, tgt, index_gap = read_graph(dataA_file, dataB_file)

    # mask掉test集的边
    src, tgt = mask_edge(testA, testB, src, tgt, index_gap)

    # 创建整图和分场景子图
    # 提取所需要的特征,I需要按照图的ReID进行排序
    u_info = torch.from_numpy(data_uinfo).to(device)

    data_iA_graph_fea = torch.from_numpy(dataA_item_info).to(device)
    data_iB_graph_fea = torch.from_numpy(dataB_item_info).to(device)

    graph, _ = create_graph(src, tgt, u_info, data_iA_graph_fea, data_iB_graph_fea, device=device)
    sub_graph_src = dgl.edge_type_subgraph(graph, [('u', 'src', 'i'), ('i', 'src-by', 'u')])
    sub_graph_tgt = dgl.edge_type_subgraph(graph, [('u', 'tgt', 'i'), ('i', 'tgt-by', 'u')])

    # 分场景获取2跳邻居
    neighbors_utable_src, neighbors_utable_tgt = neighbor_table(sub_graph_src, sub_graph_tgt, type='u')
    neighbors_itable_src, neighbors_itable_tgt = neighbor_table(sub_graph_src, sub_graph_tgt, type='i')

    # 整图的二跳子图
    graph_u_twohop = graph_hop_two(graph, type='u', device=device) # dense_to_sparse要高一下返回值data
    graph_i_twohop = graph_hop_two(graph, type='i', device=device)

    # graph_u_twohop = graph_hop_sep(graph, type='u', device=device) # dense_to_sparse要高一下返回值data
    # graph_i_twohop = graph_hop_sep(graph, type='i', device=device)


    # 图网络定义
    net = GraphNet().to(device)  

    # 
    feature_u = graph_u_twohop.ndata['info']
    feature_i = graph_i_twohop.ndata['info']
    u_feature = net(graph_u_twohop, feature_u)  # u*H
    i_feature = net(graph_i_twohop, feature_i)  # i*H

    # 图表征ID转换
    # 插入embed
    # i_feature = i_feature.cpu().detach().numpy()
    src_u_emb = nn.Embedding(dataset_A.shape[0], opt.KSIZE)  # 这里规模需要注意
    tgt_u_emb = nn.Embedding(dataset_B.shape[0], opt.KSIZE)

    src_i_emb = nn.Embedding(dataset_A.shape[1], opt.KSIZE)  # item
    tgt_i_emb = nn.Embedding(dataset_B.shape[1], opt.KSIZE)  # B item


    # 将图的表征按序存储
    src_u = torch.matmul(neighbors_utable_src.to(device), u_feature)  # D,32
    tgt_u = torch.matmul(neighbors_utable_tgt.to(device), u_feature)
    src_item = i_feature[:index_gap+1,:]
    tgt_item = i_feature[index_gap+1:,:]

    src_u_emb.weight.data = nn.Parameter(src_u)
    tgt_u_emb.weight.data = nn.Parameter(tgt_u)
    src_i_emb.weight.data = nn.Parameter(src_item)
    tgt_i_emb.weight.data = nn.Parameter(tgt_item)

    # 图有的数据
    graph_data = (src_u_emb, tgt_u_emb, src_i_emb.to(device), tgt_i_emb.to(device))
    neighbour_table = (neighbors_utable_src, neighbors_utable_tgt, neighbors_itable_src, neighbors_itable_tgt)
    return graph_data, neighbour_table

def get_douban_graph(dataset_A,dataset_B,testA, testB,data_uinfo,dataA_item_info,dataB_item_info,dataA_overlap,dataB_overlap,device):
    # 下面是图的数据，需要对齐
    dataA_file = opt.DATAA + '/ratings_p.csv'  # 注意转换
    dataB_file = opt.DATAB + '/ratings_p.csv'

    src, tgt, index_gap = read_graph(dataA_file, dataB_file)  # ID_REID

    # mask掉test集的边
    src, tgt = mask_edge(testA, testB, src, tgt, index_gap)

    if opt.ABA:
        # mask掉overlap的边
        src,tgt = mask_edge(dataA_overlap, dataB_overlap, src, tgt, index_gap)

    # 创建整图和分场景子图
    # 提取所需要的特征,I需要按照图的ReID进行排序
    u_info = torch.from_numpy(data_uinfo).to(device)

    data_iA_graph_fea = torch.from_numpy(dataA_item_info).to(device)
    data_iB_graph_fea = torch.from_numpy(dataB_item_info).to(device)

    graph, _ = create_graph(src, tgt, u_info, data_iA_graph_fea, data_iB_graph_fea, device=device)
    sub_graph_src = dgl.edge_type_subgraph(graph, [('u', 'src', 'i'), ('i', 'src-by', 'u')])
    sub_graph_tgt = dgl.edge_type_subgraph(graph, [('u', 'tgt', 'i'), ('i', 'tgt-by', 'u')])

    # 分场景获取2跳邻居
    neighbors_utable_src, neighbors_utable_tgt = neighbor_table(sub_graph_src, sub_graph_tgt, type='u')
    # neighbors_itable_src, neighbors_itable_tgt = neighbor_table(sub_graph_src, sub_graph_tgt, type='i')
    neighbors_itable_src, neighbors_itable_tgt = [],[]
    # 图网络定义
    net = GraphNet_NGCF(graph, opt.KSIZE, [opt.KSIZE]*opt.GCN_LAYER, [0.1]*opt.GCN_LAYER).to(device)  # 后续i也可以搞一个单独的GCN，或者直接一个RGCN直接卷，忽视所有的点。

    u_feature, i_feature = net(graph,'u', 'i')  # u*H

    # 图表征ID转换
    # 插入embed
    # i_feature = i_feature.cpu().detach().numpy()
    src_u_emb = nn.Embedding(dataset_A.shape[0], opt.KSIZE)  # 这里规模需要注意
    tgt_u_emb = nn.Embedding(dataset_B.shape[0], opt.KSIZE)

    src_i_emb = nn.Embedding(dataset_A.shape[1], opt.KSIZE)  # item
    tgt_i_emb = nn.Embedding(dataset_B.shape[1], opt.KSIZE)  # B item

    # 这里转为邻接矩阵
    neighbors_utable_src, src_edge_weight = dense_to_sparse(neighbors_utable_src)
    neighbors_utable_tgt, tgt_edge_weight = dense_to_sparse(neighbors_utable_tgt)
    src_u,tgt_u = aggravate_domain(neighbors_utable_src, neighbors_utable_tgt, u_feature,device)
    print("A",src_u.shape)

    # 将图的表征按序存储
    # src_u = torch.matmul(neighbors_utable_src.to(device), u_feature)  # D,32
    # tgt_u = torch.matmul(neighbors_utable_tgt.to(device), u_feature)
    # print("B",src_u.shape)

    src_item = i_feature[:index_gap+1,:]
    tgt_item = i_feature[index_gap+1:,:]

    src_u_emb.weight.data = nn.Parameter(src_u)
    tgt_u_emb.weight.data = nn.Parameter(tgt_u)
    src_i_emb.weight.data = nn.Parameter(src_item)
    tgt_i_emb.weight.data = nn.Parameter(tgt_item)

    # 图有的数据
    graph_data = (src_u_emb, tgt_u_emb, src_i_emb.to(device), tgt_i_emb.to(device))
    neighbour_table = (neighbors_utable_src, neighbors_utable_tgt, neighbors_itable_src, neighbors_itable_tgt)
    return graph_data, neighbour_table


def get_douban_data(device):
    # 数据读取
    # 图表征，随机
    # model_N2V_A = torch.randn(15240, 32).to(device)  # user_size+item_size,H
    # model_N2V_B = torch.randn(15240, 32).to(device)

    # 数据源读取
    Ksize = opt.KSIZE
    dataName_A = opt.DATAA.split(sep="/")[-1]
    dataName_B = opt.DATAB.split(sep="/")[-1]

    # 交互数据
    dataset_A, dataset_B = Dataset(opt.DATAA), Dataset(opt.DATAB)

    # u.i feature
    # data_uinfo = np.random.randn(2718,32).astype(np.float32) # tensor中均为float32类型
    # dataA_item_info = np.random.randn(9565,32).astype(np.float32) 
    # dataB_item_info = np.random.randn(6777,32).astype(np.float32)

    feature_A = Doc2Vec.load(
        (opt.DATAA + "/Doc2vec_" + dataName_A + "_VSize%02d" + ".model") % opt.KSIZE).docvecs.vectors_docs
    feature_B = Doc2Vec.load(
        (opt.DATAB + "/Doc2vec_" + dataName_B + "_VSize%02d" + ".model") % opt.KSIZE).docvecs.vectors_docs

    dataA_uinfo, dataA_item_info = feature_A[:dataset_A.shape[0], :], feature_A[dataset_A.shape[0]:,
                                                                      :]  # u和I的内部顺序需要查看一下
    dataB_uinfo, dataB_item_info = feature_B[:dataset_B.shape[0], :], feature_B[dataset_B.shape[0]:, :]

    # I 内部顺序重排,lenovo数据重排好了
    index_A = (pd.read_csv(opt.DATAA + '/' + dataName_A.split('_')[-1] + "_feature_p.csv")['UID'].values - 1).tolist()
    index_B = (pd.read_csv(opt.DATAB + '/' + dataName_B.split('_')[-1] + "_feature_p.csv")['UID'].values - 1).tolist()
    mapA_table = dict(zip(index_A, list(range(len(index_A)))))
    index_A_r = np.array(sorted(mapA_table.items(), key=operator.itemgetter(0)))[:, 1].tolist()
    mapB_table = dict(zip(index_B, list(range(len(index_B)))))
    index_B_r = np.array(sorted(mapB_table.items(), key=operator.itemgetter(0)))[:, 1].tolist() # 字典排序反转
    dataA_item_info = np.take(dataA_item_info, index_A_r, axis=0) # 按照index重排序,这里是取行
    dataB_item_info = np.take(dataB_item_info, index_B_r, axis=0)


    data_uinfo = np.maximum(dataA_uinfo,dataB_uinfo)

    o_src_u_emb = nn.Embedding(dataset_A.shape[0], opt.KSIZE)
    o_tgt_u_emb = nn.Embedding(dataset_B.shape[0], opt.KSIZE)
    o_src_i_emb = nn.Embedding(dataset_A.shape[1], opt.KSIZE)
    o_tgt_i_emb = nn.Embedding(dataset_B.shape[1], opt.KSIZE)
    o_src_u_emb.weight.data = nn.Parameter(torch.from_numpy(dataA_uinfo).to(device))
    o_tgt_u_emb.weight.data = nn.Parameter(torch.from_numpy(dataB_uinfo).to(device))
    o_src_i_emb.weight.data = nn.Parameter(torch.from_numpy(dataA_item_info).to(device))
    o_tgt_i_emb.weight.data = nn.Parameter(torch.from_numpy(dataB_item_info).to(device))
    origin_emb = [o_src_u_emb, o_tgt_u_emb, o_src_i_emb, o_tgt_i_emb]
    return origin_emb,dataset_A, dataset_B, data_uinfo,dataA_item_info,dataB_item_info,dataName_A,dataName_B


def get_lenovo_data(device):
    # 数据源读取
    Ksize = opt.KSIZE
    dataName_A = opt.DATAA.split(sep="/")[-1]
    dataName_B = opt.DATAB.split(sep="/")[-1]

    # 交互数据
    dataset_A, dataset_B = Dataset(opt.DATAA), Dataset(opt.DATAB)

    feature_A = Doc2Vec.load(
        (opt.DATAA + "/Doc2vec_" + dataName_A + "_VSize%02d" + ".model") % opt.KSIZE).docvecs.vectors_docs
    feature_B = Doc2Vec.load(
        (opt.DATAB + "/Doc2vec_" + dataName_B + "_VSize%02d" + ".model") % opt.KSIZE).docvecs.vectors_docs

    dataA_uinfo, dataA_item_info = feature_A[:dataset_A.shape[0], :], feature_A[dataset_A.shape[0]:,
                                                                      :]  # u和I的内部顺序需要查看一下
    dataB_uinfo, dataB_item_info = feature_B[:dataset_B.shape[0], :], feature_B[dataset_B.shape[0]:, :]


    data_uinfo = np.maximum(dataA_uinfo,dataB_uinfo)

    o_src_u_emb = nn.Embedding(dataset_A.shape[0], opt.KSIZE)
    o_tgt_u_emb = nn.Embedding(dataset_B.shape[0], opt.KSIZE)
    o_src_i_emb = nn.Embedding(dataset_A.shape[1], opt.KSIZE)
    o_tgt_i_emb = nn.Embedding(dataset_B.shape[1], opt.KSIZE)
    o_src_u_emb.weight.data = nn.Parameter(torch.from_numpy(dataA_uinfo).to(device))
    o_tgt_u_emb.weight.data = nn.Parameter(torch.from_numpy(dataB_uinfo).to(device))
    o_src_i_emb.weight.data = nn.Parameter(torch.from_numpy(dataA_item_info).to(device))
    o_tgt_i_emb.weight.data = nn.Parameter(torch.from_numpy(dataB_item_info).to(device))
    origin_emb = [o_src_u_emb, o_tgt_u_emb, o_src_i_emb, o_tgt_i_emb]
    return origin_emb,dataset_A, dataset_B, data_uinfo,dataA_item_info,dataB_item_info,dataName_A,dataName_B


def get_lenovo_graph(dataset_A,dataset_B,testA, testB,data_uinfo,dataA_item_info,dataB_item_info,dataA_overlap,dataB_overlap,device):
    # 下面是图的数据，需要对齐
    dataA_file = opt.DATAA + '/ratings_p.csv'  # 注意转换
    dataB_file = opt.DATAB + '/ratings_p.csv'

    src, tgt, index_gap = read_graph(dataA_file, dataB_file)  # ID_REID

    # mask掉test集的边
    src, tgt = mask_edge(testA, testB, src, tgt, index_gap)

    if opt.ABA:
        # mask掉overlap的边
        src,tgt = mask_edge(dataA_overlap, dataB_overlap, src, tgt, index_gap)

    # 创建整图和分场景子图
    # 提取所需要的特征,I需要按照图的ReID进行排序
    u_info = torch.from_numpy(data_uinfo).to(device)

    data_iA_graph_fea = torch.from_numpy(dataA_item_info).to(device)
    data_iB_graph_fea = torch.from_numpy(dataB_item_info).to(device)

    graph, _ = create_graph(src, tgt, u_info, data_iA_graph_fea, data_iB_graph_fea, device=device)
    sub_graph_src = dgl.edge_type_subgraph(graph, [('u', 'src', 'i'), ('i', 'src-by', 'u')])
    sub_graph_tgt = dgl.edge_type_subgraph(graph, [('u', 'tgt', 'i'), ('i', 'tgt-by', 'u')])

    # 分场景获取2跳邻居
    neighbors_utable_src, neighbors_utable_tgt = neighbor_table(sub_graph_src, sub_graph_tgt, type='u') # 6w用户个，6w*6w
    # neighbors_itable_src, neighbors_itable_tgt = neighbor_table(sub_graph_src, sub_graph_tgt, type='i')
    neighbors_itable_src, neighbors_itable_tgt = [],[]

    # 图网络定义 0.1
    net = GraphNet_NGCF(graph, opt.KSIZE, [opt.KSIZE]*opt.GCN_LAYER, [0.1]*opt.GCN_LAYER).to(device)  # 后续i也可以搞一个单独的GCN，或者直接一个RGCN直接卷，忽视所有的点。

    u_feature, i_feature = net(graph,'u', 'i')  # u*H

    # 图表征ID转换
    # 插入embed
    # i_feature = i_feature.cpu().detach().numpy()
    src_u_emb = nn.Embedding(dataset_A.shape[0], opt.KSIZE)  
    tgt_u_emb = nn.Embedding(dataset_B.shape[0], opt.KSIZE)

    src_i_emb = nn.Embedding(dataset_A.shape[1], opt.KSIZE)  # item
    tgt_i_emb = nn.Embedding(dataset_B.shape[1], opt.KSIZE)  # B item


    # 这里转为邻接矩阵
    neighbors_utable_src, src_edge_weight = dense_to_sparse(neighbors_utable_src)
    neighbors_utable_tgt, tgt_edge_weight = dense_to_sparse(neighbors_utable_tgt)
    src_u,tgt_u = aggravate_domain(neighbors_utable_src, neighbors_utable_tgt, u_feature,device)
    # print(src_u.shape)

    # 将图的表征按序存储
    # src_u = torch.matmul(neighbors_utable_src.to(device), u_feature)  # D,32
    # tgt_u = torch.matmul(neighbors_utable_tgt.to(device), u_feature)
    src_item = i_feature[:index_gap+1,:]
    tgt_item = i_feature[index_gap+1:,:]

    src_u_emb.weight.data = nn.Parameter(src_u)
    tgt_u_emb.weight.data = nn.Parameter(tgt_u)
    src_i_emb.weight.data = nn.Parameter(src_item)
    tgt_i_emb.weight.data = nn.Parameter(tgt_item)

    # 图有的数据
    graph_data = (src_u_emb, tgt_u_emb, src_i_emb.to(device), tgt_i_emb.to(device))
    neighbour_table = (neighbors_utable_src, neighbors_utable_tgt, neighbors_itable_src, neighbors_itable_tgt)
    return graph_data, neighbour_table



def train_test_split(dataset_A,dataset_B):
    train_A, testA = dataset_A.train, dataset_A.test
    train_B, testB = dataset_B.train, dataset_B.test
    testNegA, testNegB = dataset_A.getTestNeg(testA, 99), dataset_B.getTestNeg(testB, 99)

    train_u_A, train_i_A, train_r_A = dataset_A.getInstance(
        train_A, opt.NEGNUM)  # array,array,array Dataset *(1+negnum)
    train_len_A = len(train_u_A)
    shuffled_idx_A = np.random.permutation(np.arange(train_len_A))
    train_u_A = train_u_A[shuffled_idx_A]
    train_i_A = train_i_A[shuffled_idx_A]
    train_r_A = train_r_A[shuffled_idx_A]

    train_u_B, train_i_B, train_r_B = dataset_B.getInstance(
        train_B, opt.NEGNUM)
    train_len_B = len(train_u_B)
    shuffled_idx_B = np.random.permutation(np.arange(train_len_B))
    train_u_B = train_u_B[shuffled_idx_B]
    train_i_B = train_i_B[shuffled_idx_B]
    train_r_B = train_r_B[shuffled_idx_B]

    # 这里用作mask消融实验
    dataA_overlap = []
    dataB_overlap = []
    if opt.ABA:
        overlap = list(set(train_u_A[train_r_A!=0]).intersection(set(train_u_B[[train_r_B!=0]]))) # 
        overlap_mask = overlap[:round(len(overlap)*opt.Ratio)] 
        train_u_A,train_i_A,train_r_A,train_len_A,dataA_overlap = mask_data(train_u_A,train_i_A,train_r_A,overlap_mask)
        train_u_B,train_i_B,train_r_B,train_len_B,dataB_overlap = mask_data(train_u_B,train_i_B,train_r_B,overlap_mask)


    return train_u_A, train_i_A, train_r_A ,train_u_B,train_i_B,train_r_B, testNegA, testNegB,testA,testB,train_len_A,train_len_B,dataA_overlap,dataB_overlap



def train(**kwargs):
    opt.parse(kwargs)
    # GPU
    device = 'cpu'
    use_cuda = opt.GPU_USED
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    # 模型定义
    model = Contra_Two_Tower(input_size=opt.KSIZE).to(device)
    model.optimizer = torch.optim.Adam(model.parameters(), lr=opt.LR, weight_decay=opt.WEIGHT_DECAY)
    model.criterion = criterion

    # 数据
    origin_emb, dataset_A, dataset_B, data_uinfo, dataA_item_info, dataB_item_info,dataName_A,dataName_B = get_lenovo_data(device)
    inter_embed_dict = add_embedding_matrix(dataset_A, dataset_B) # 传统的MF
    data_info = {"shape_A":dataset_A.shape,"shape_B":dataset_B.shape,
                 "maxRate_A":dataset_A.maxRate, "maxRate_B":dataset_B.maxRate}

    # 数据划分
    train_u_A, train_i_A, train_r_A, train_u_B, train_i_B, train_r_B,testNegA, testNegB,testA,testB,train_len_A,train_len_B,dataA_overlap,dataB_overlap = train_test_split(dataset_A,dataset_B)

    # 图数据获取
    graph_data, neighbour_table = get_lenovo_graph(dataset_A,dataset_B,testA, testB,data_uinfo,dataA_item_info,dataB_item_info,dataA_overlap,dataB_overlap,device)
    # graph_data, neighbour_table = get_douban_graph_origin(dataset_A,dataset_B,testA, testB,data_uinfo,dataA_item_info,dataB_item_info,device)

    # 指标设计
    num_batches_A = len(train_u_A) // opt.BATCH_SIZE + 1
    num_batches_B = len(train_u_B) // opt.BATCH_SIZE + 1

    best_hr_A = -1
    best_NDCG_A = -1
    best_epoch_A = -1
    best_hr_B = -1
    best_NDCG_B = -1
    best_epoch_B = -1
    allResults_A = []
    allResults_B = []
    print("Start Training!")
    # 开始训练
    cur_iter = 0
    for epoch in range(opt.EPOCHS):
        if epoch < opt.START:
            pretrain = True
        else:
            pretrain = False
        losses_A = []
        losses_B = []
        max_num_batches = max(num_batches_A, num_batches_B)  # max batch
        print("max_batch",max_num_batches)

        print("=" * 20 + "Epoch ", epoch, "=" * 20)
        for i in range(max_num_batches):
            cur_iter += 1
            # 选择数据,同样维护两套emb, joint emb
            min_idx = i * opt.BATCH_SIZE
            max_idx_A = np.min([train_len_A, (i + 1) * opt.BATCH_SIZE])
            max_idx_B = np.min([train_len_B, (i + 1) * opt.BATCH_SIZE])
            if min_idx < train_len_A:  # the training for domain A has not completed
                train_u_batch_A = train_u_A[min_idx:max_idx_A] # [4096]
                train_i_batch_A = train_i_A[min_idx:max_idx_A]
                train_r_batch_A = train_r_A[min_idx:max_idx_A]

                feed_dict_A = create_feed_dict(train_u_batch_A,
                                                    train_i_batch_A, 'A',
                                                    train_r_batch_A, device=device)
                tmp_loss_A, _y_A = train_step(cur_iter, model, feed_dict_A, inter_embed_dict, graph_data,origin_emb, neighbour_table, data_info,pretrain,"A")
                losses_A.append(tmp_loss_A)

            if min_idx < train_len_B:  # the training for domain B has not completed
                train_u_batch_B = train_u_B[min_idx:max_idx_B]
                train_i_batch_B = train_i_B[min_idx:max_idx_B]
                train_r_batch_B = train_r_B[min_idx:max_idx_B]
                feed_dict_B = create_feed_dict(train_u_batch_B,
                                                    train_i_batch_B, 'B',
                                                    train_r_batch_B, device=device)
                tmp_loss_B, _y_B = train_step(cur_iter, model, feed_dict_B,inter_embed_dict, graph_data,origin_emb, neighbour_table, data_info,pretrain,"B")
                losses_B.append(tmp_loss_B)

            # if opt.LOG_STEP_FREQ and i % opt.LOG_STEP_FREQ ==0:
            #     # 查看最后一个batch的mean
            #     sys.stdout.write('\r{} / {} : loss = {};'.format(
            #         i, max_num_batches, np.mean(losses_A[-opt.LOG_STEP_FREQ:])))
            #     sys.stdout.write('\r{} / {} : loss = {}'.format(
            #         i, max_num_batches, np.mean(losses_B[-opt.LOG_STEP_FREQ:])))
            #     sys.stdout.flush()
        loss_A = np.mean(losses_A)
        loss_B = np.mean(losses_B)
        print("\nMean loss in this epoch is: Domain A={};Domain B={}".format(
            loss_A, loss_B))

        print('=' * 50)
        print("Start Evaluation!")
        topK = opt.TOPK

        hr_A, NDCG_A, hr_B, NDCG_B = evaluate(model,testNegA,testNegB,graph_data,origin_emb,neighbour_table,data_info,device)
        allResults_A.append([epoch, topK, hr_A, NDCG_A])
        allResults_B.append([epoch, topK, hr_B, NDCG_B])
        print(
            "Epoch ", epoch,
            "Domain A: {} TopK: {} HR: {}, NDCG: {}".format(
                dataName_A, topK, hr_A, NDCG_A))
        print(
            "Epoch ", epoch,
            "Domain B: {} TopK: {} HR: {}, NDCG: {}".format(
                dataName_B, topK, hr_B, NDCG_B))
        if hr_A > best_hr_A:
            best_hr_A = hr_A
            best_epoch_A = epoch
        if NDCG_A > best_NDCG_A:
            best_NDCG_A = NDCG_A
        if hr_B > best_hr_B:
            best_hr_B = hr_B
            best_epoch_B = epoch
        if NDCG_B > best_NDCG_B:
            best_NDCG_B = NDCG_B
        print("=" * 20 + "Epoch ", epoch, "End" + "=" * 20)
    print(
        "Domain A: Best hr: {}, NDCG: {}, At Epoch {}; Domain B: Best hr: {}, NDCG: {}, At Epoch {}"
            .format(best_hr_A, best_NDCG_A, best_epoch_A, best_hr_B,
                    best_NDCG_B, best_epoch_B))  
    bestPerformance = [[best_hr_A, best_NDCG_A, best_epoch_A],
                       [best_hr_B, best_NDCG_B, best_epoch_B]]
    matname = opt.LOGFILE + 'CrossRec_' + str(dataName_A) + '_' + str(
        dataName_B) + '_KSize_' + str(opt.KSIZE) + '_Result.mat'
    scio.savemat(
        matname, {
            'allResults_A': allResults_A,
            'allResults_B': allResults_B,
            'bestPerformance': bestPerformance
        })
    print("Training complete!")

if __name__ == '__main__':
    print("Hello Cross Domain !")
    train()

