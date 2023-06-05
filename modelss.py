#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/24 22:15
# @Author  : Jack Zhao
# @Site    : 
# @File    : models.py
# @Software: PyCharm

# #Desc: 
import torch
import dgl
import math
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import dgl.function as fn

from torch.autograd import Variable
# from dgl.nn import GraphConv
from utils import activation_layer,l2_loss,sinkhorn,reg
from config import opt



class Contra_Two_Tower(nn.Module):
    def __init__(self, input_size):
        super(Contra_Two_Tower, self).__init__()
        # 线性变换
        self.user_W1_A = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(2*input_size, opt.KSIZE)))  
        self.user_W1_B = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(2*input_size, opt.KSIZE)))
        self.item_W1_A = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(2*input_size, opt.KSIZE)))
        self.item_W1_B = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(2*input_size, opt.KSIZE))) 


        self.A_B_transfer = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(opt.KSIZE, opt.KSIZE)))  # empty有问题
        self.B_A_transfer = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(opt.KSIZE, opt.KSIZE)))  # empty有问题


        # swav z q
        self.swav = Swav()

        # Tower
        self.item_A_dnn = Dnn(input_size,[opt.KSIZE, 2 * opt.KSIZE, 4 * opt.KSIZE, 8 * opt.KSIZE,
                            4 * opt.KSIZE, 2 * opt.KSIZE, opt.KSIZE],'leakyrelu',0.,False) 
        self.item_B_dnn = Dnn(input_size,[opt.KSIZE, 2 * opt.KSIZE, 4 * opt.KSIZE, 8 * opt.KSIZE,
                            4 * opt.KSIZE, 2 * opt.KSIZE, opt.KSIZE],'leakyrelu',0.,False)
        self.user_A_dnn = Dnn(2*input_size, [opt.KSIZE, 2 * opt.KSIZE, 4 * opt.KSIZE, 8 * opt.KSIZE,
                                         4 * opt.KSIZE, 2 * opt.KSIZE, opt.KSIZE],'leakyrelu',0.,False)
        self.user_B_dnn = Dnn(2*input_size, [opt.KSIZE, 2 * opt.KSIZE, 4 * opt.KSIZE, 8 * opt.KSIZE,
                                         4 * opt.KSIZE, 2 * opt.KSIZE, opt.KSIZE], 'leakyrelu',0.,False)

        self.init_weights()
        # self.apply(self._init_weights)


    def init_weights(self):
        """尝试换换"""
        stdv = 1.0 / math.sqrt(opt.KSIZE)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def _init_weights(self,module):
        stdv = 1.0 / math.sqrt(opt.KSIZE)
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-stdv,stdv)
            if module.bias is not None:
                module.bias.data.zero_()


    def forward(self,user_input_A,user_input_B,item_input_A,item_input_B):
        # linear transfermation
        user_out_A_ = torch.matmul(user_input_A, self.user_W1_A)  # B,Ksize
        user_out_B_ = torch.matmul(user_input_B, self.user_W1_B)

        item_out_A_ = torch.matmul(item_input_A, self.item_W1_A)
        item_out_B_ = torch.matmul(item_input_B, self.item_W1_B)
        # user_out_A_ = self.user_W1_A(user_input_A) # B,Ksize
        # user_out_B_ = self.user_W1_B(user_input_B)
        #
        # item_out_A_ = self.item_W1_A(item_input_A)
        # item_out_B_ = self.item_W1_B(item_input_B)

        # print("F", torch.isnan(self.user_W1_A.data).any(),torch.isnan(self.user_W1_B.data).any(),torch.isnan(self.item_W1_A.data).any(),torch.isnan(self.item_W1_B.data).any())

        # swav loss
        # normalize the prototypes
        with torch.no_grad():
            w = self.swav.prototypes.weight.data.clone()
            w = nn.functional.normalize(w,dim=1,p=2)
            self.swav.prototypes.weight.copy_(w)

        user_z_A, user_q_A = self.swav(user_out_A_) 
        user_z_B, user_q_B = self.swav(user_out_B_)
        # item_z_A, item_q_A = self.swav(item_out_A)
        # item_z_B, item_q_B = self.swav(item_out_B)

        user_z_A, user_z_B = user_z_A.detach(),user_z_B.detach() #,item_z_A.detach(),item_z_B.detach()

        user_q_A_, user_q_B_ = user_q_A.detach(),user_q_B.detach()

        # print(user_q_B.requires_grad)
        with torch.no_grad():
            user_q_A_, user_q_B_ = sinkhorn(user_q_A_), sinkhorn(user_q_B_) 

        user_q_A = user_q_A /opt.TEMPERATURE
        # print(user_q_B.requires_grad)
        user_swav_A = -torch.sum(torch.sum(user_q_B_ * F.log_softmax(user_q_A, dim=1), dim=1)) 
        user_q_B = user_q_B / opt.TEMPERATURE
        user_swav_B = -torch.sum(torch.sum(user_q_A_ * F.log_softmax(user_q_B, dim=1), dim=1))

        # user_swav = (user_swav_A+user_swav_B)/2

        # item_q_A = item_q_A / opt.TEMPERATURE
        # item_swav_A = -torch.mean(torch.sum(item_q_B * F.log_softmax(item_q_A, dim=1), dim=1)) 
        # item_q_B = item_q_B / opt.TEMPERATURE
        # item_swav_B = -torch.mean(torch.sum(item_q_A * F.log_softmax(item_q_B, dim=1), dim=1))
        # item_swav = (item_swav_A + item_swav_B) / 2 # 和emb关系不大

        # print("B",torch.isnan(user_out_A_).any(),torch.isnan(user_out_B_).any(),torch.isnan(item_out_A_).any(),torch.isnan(item_out_B_).any())


        # 仿照GADTCDR
        user_out_A_C = torch.concat((
            user_out_A_,torch.matmul(user_out_B_,self.B_A_transfer)
        ),dim=-1)
        user_out_B_C = torch.concat((
            user_out_A_, torch.matmul(user_out_A_, self.A_B_transfer)
        ),dim=-1)

        # two tower
        user_out_A,user_out_B = self.user_A_dnn(user_out_A_C),self.user_B_dnn(user_out_B_C)
        item_out_A,item_out_B = self.item_A_dnn(item_out_A_),self.item_B_dnn(item_out_B_)


        # print("C",torch.isnan(user_out_A).any(),torch.isnan(user_out_B).any(),torch.isnan(item_out_A).any(),torch.isnan(item_out_B).any())

        # reg
        norm_user_output_A = reg(user_out_A)
        norm_item_output_A = reg(item_out_A)
        norm_user_output_B = reg(user_out_B)
        norm_item_output_B = reg(item_out_B)

        regularizer_A = l2_loss(user_out_A) + l2_loss(
            item_out_A)
        regularizer_B = l2_loss(user_out_B) + l2_loss(
            item_out_B)


        # dot
        y_A = torch.sum(
            torch.multiply(user_out_A, item_out_A), axis=1,
            keepdims=False) / (norm_item_output_A * norm_user_output_A)
        y_A = torch.clamp(y_A,1e-6)
        y_B = torch.sum(
            torch.multiply(user_out_B, item_out_B), axis=1,
            keepdims=False) / (norm_item_output_B * norm_user_output_B)
        y_B = torch.clamp(y_B,1e-6)  # 最小为1e-6


        return regularizer_A, regularizer_B, user_swav_A, user_swav_B, y_A, y_B, \
               (user_out_A_C, item_out_A, user_out_B_C, item_out_B,norm_item_output_A,norm_item_output_B)






class Swav(nn.Module):
    def __init__(self):
        super(Swav, self).__init__()
        self.emb = Dnn(opt.KSIZE, [opt.KSIZE, 2 * opt.KSIZE], 'relu')
        self.projection = Dnn(2 * opt.KSIZE, [opt.KSIZE], 'relu')
        self.prototypes = nn.Linear(opt.KSIZE, opt.PROTONUM, bias=False)  


    def forward_head(self,x):
        x = self.projection(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        prototypes = self.prototypes(x) 
        return x, prototypes


    def forward(self,input):
        output = self.emb(input)
        return self.forward_head(output)
    

class Dnn(nn.Module):
    """Dnn(input_size*2,[100,50,1],'sigmoid')"""
    def __init__(self, input_size,hidden_units,activation='relu',dropout_rate=0.,use_bn=False,dice_dim=3):
        super(Dnn, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("Hidden units is empty, list needed!")
        hidden_units = [input_size] + list(hidden_units)
        self.linears = nn.ModuleList([nn.Linear(hidden_units[i],hidden_units[i+1]) for i in range(len(hidden_units)-1)])
        if self.use_bn:
            self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_units[i+1]) for i in range(len(hidden_units)-1)])
        self.activation_layers = nn.ModuleList([activation_layer(activation, hidden_units[i + 1], dice_dim) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(tensor)
                # nn.init.normal_(tensor, mean=0, std=0.001)
                # nn.init.trunc_normal_(tensor,mean=0,std=0.001)

    def forward(self, inputs):
        deep_input = inputs
        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)
            if self.use_bn:
                fc = self.bn[i](fc)
            fc = self.activation_layers[i](fc) # 激活
            fc = self.dropout(fc)
            deep_input = fc
        return deep_input

#
# def read_graph_origin(src_file, tgt_file):
#     """读取graph数据,这里因为要合并item侧"""
#     # src = pd.read_csv(src_file, sep='\t', header=None, index_col=None)
#     # tgt = pd.read_csv(tgt_file, sep='\t', header=None, index_col=None)
#     src = pd.read_csv(src_file, header=0, index_col=0) # 测试
#     tgt = pd.read_csv(tgt_file, header=0, index_col=0)
#     src.columns = ['u', 'i', 'r']
#     tgt.columns = ['u', 'i', 'r']
#     # 统一u, 这个u已经从0开始了
#     src['u'] = src['u']-1
#     tgt['u'] = tgt['u']-1
#     src['i'],tgt['i'] = src['i']-1,tgt['i']-1
#
#     # max_item_id = len(src['i'].unique()) + len(tgt['i'].unique())
#     # print(max_user_id,max(src['i'].unique()),max(tgt['i'].unique()))
#
#     # max_user_id = max(max(src['u']), max(tgt['u']))
#     print("A",len(src['u'].unique()),"B",len(tgt["u"].unique())) # 重叠率
#
#     # index = max_user_id + 1
#     item_src_map = {}
#     for index, id in enumerate(src["i"].unique(),start=0):
#         item_src_map[str(index)] = str(id)
#
#     item_tgt_map = {}
#     print("End_index", index)
#     index = index + 1
#     for id in tgt['i'].unique():
#         item_tgt_map[str(index)] = str(id)
#         index +=1
#     item_src_map = dict(zip(item_src_map.values(), item_src_map.keys()))
#     item_tgt_map = dict(zip(item_tgt_map.values(), item_tgt_map.keys()))
#
#     src['i'] = src['i'].apply(lambda x: item_src_map[str(x)])
#     tgt['i'] = tgt['i'].apply(lambda x: item_tgt_map[str(x)])
#
#     src['i'],tgt['i'] = src['i'].astype(int),tgt['i'].astype(int)
#
#     return src, tgt, item_src_map, item_tgt_map


def read_graph(src_file, tgt_file):
    """读取graph数据,这里因为要合并item侧"""
    # src = pd.read_csv(src_file, sep='\t', header=None, index_col=None)
    # tgt = pd.read_csv(tgt_file, sep='\t', header=None, index_col=None)
    src = pd.read_csv(src_file, header=0, index_col=0) # 测试
    tgt = pd.read_csv(tgt_file, header=0, index_col=0)
    src.columns = ['u', 'i', 'r']
    tgt.columns = ['u', 'i', 'r']
    # 统一u, 这个u已经从0开始了
    src['u'] = src['u']-1
    tgt['u'] = tgt['u']-1
    src['i'] = src['i']-1
    tgt['i'] = max(src['i']) + tgt['i']

    print("A",len(src['u'].unique()),"B",len(tgt["u"].unique())) # 重叠率
    return src, tgt,max(src['i'])



def create_graph(src,tgt,u_info,src_info,tgt_info,device):
    """创建图"""
    src_start = torch.from_numpy(src['u'].values)
    src_end = torch.from_numpy(src['i'].values)
    tgt_start = torch.from_numpy(tgt['u'].values)
    tgt_end = torch.from_numpy(tgt['i'].values)


    graph_data = {
        ('u','src','i'): (src_start, src_end),
        ('i', 'src-by', 'u'): (src_end, src_start),
        ('u','tgt','i'): (tgt_start,tgt_end),
        ('i', 'tgt-by', 'u'): (tgt_end, tgt_start)
    } # i只是关系不同

    graph = dgl.heterograph(graph_data).to(device)

    i_info = torch.cat((src_info, tgt_info),axis=0)
    graph.nodes['u'].data['info'] = u_info
    graph.nodes['i'].data['info'] = i_info

    # 转为无向图
    print("Graph Has Done !")
    print(graph)
    print("Original order",graph.ntypes)

    # 转成同构图
    # graph_homo = dgl.to_homogeneous(graph, ndata=['info']) # reorder
    graph_homo = ""
    # print("Corr NType",graph_homo.ndata[dgl.NTYPE])
    # print("Corr NID",graph_homo.ndata[dgl.NID])

    return graph, graph_homo


def graph_hop(graph,hop=2):
    """获取二阶邻居子图,过慢"""
    graph = dgl.transforms.khop_graph(graph, hop, copy_ndata=True) # no reorder subgraph
    graph = dgl.remove_self_loop(graph)
    return graph


def neighbor_table(sub_graph_src,sub_graph_tgt,type='u'):
    """不同context邻居表"""
    if type == 'u':
        adj = sub_graph_src.adjacency_matrix(etype='src').to_dense()
        adj_inverse = sub_graph_src.adjacency_matrix(etype='src-by').to_dense()
        neighbors_table_src = torch.matmul(adj, adj_inverse)
        neighbors_table_src = F.normalize(neighbors_table_src, p=2, dim=1) # 归一化加权,改成稀疏矩阵乘法，sparse、再存一个图
        # neighbors_table_src = (neighbors_table_src.t() / torch.norm(neighbors_table_src, p=2, dim=1)).t() # 归一化,其实就是算邻居
        adj = sub_graph_tgt.adjacency_matrix(etype='tgt').to_dense()
        adj_inverse = sub_graph_tgt.adjacency_matrix(etype='tgt-by').to_dense()
        neighbors_table_tgt = torch.matmul(adj, adj_inverse)
        neighbors_table_tgt = F.normalize(neighbors_table_tgt, p=2, dim=1)
        # neighbors_table_tgt = (neighbors_table_tgt.t() / torch.norm(neighbors_table_tgt, p=2, dim=1)).t()
    else:
        adj = sub_graph_src.adjacency_matrix(etype='src-by').to_dense()
        adj_inverse = sub_graph_src.adjacency_matrix(etype='src').to_dense()
        neighbors_table_src = torch.matmul(adj, adj_inverse)
        neighbors_table_src = F.normalize(neighbors_table_src, p=2, dim=1)
        # neighbors_table_src = (neighbors_table_src.t() / torch.norm(neighbors_table_src, p=2, dim=1)).t()
        adj = sub_graph_tgt.adjacency_matrix(etype='tgt-by').to_dense()
        adj_inverse = sub_graph_tgt.adjacency_matrix(etype='tgt').to_dense()
        neighbors_table_tgt = torch.matmul(adj, adj_inverse)
        neighbors_table_tgt = F.normalize(neighbors_table_tgt, p=2, dim=1)
        # neighbors_table_tgt = (neighbors_table_tgt.t() / torch.norm(neighbors_table_tgt, p=2, dim=1)).t()
    return neighbors_table_src, neighbors_table_tgt


def graph_hop_sep(graph,type,device):
    """创建二跳子图"""
    if type == 'u':
        feature = graph.ndata['info']['u']
        graph_adj = dgl.edge_type_subgraph(graph,['src','src-by'])
        graph_combine = graph_adj['u', :, 'i'] # 合并边类型
        adj = graph_combine.adjacency_matrix().to_dense() # U.I
        neighbors_table = torch.matmul(adj,adj.T) # U,U
        print(adj.shape)
        src_start, src_end = dense_to_sparse(neighbors_table)
        graph_two = dgl.graph((src_start,src_end)).to(device) # # u.shape*u.shape
        graph_two.ndata['info'] = feature
    else:
        # feature = graph.ndata['info']['i']
        graph_adj = dgl.edge_type_subgraph(graph, ['tgt', 'tgt-by'])
        graph_combine = graph_adj['u', :, 'i']
        adj = graph_combine.adjacency_matrix().to_dense()

        neighbors_table = torch.matmul(adj,adj.T) # I.shape*I.shape
        src_start, src_end = dense_to_sparse(neighbors_table)
        graph_two = dgl.graph((src_start, src_end)).to(device)
        # graph_two.ndata['info'] = feature
    return graph_two


def graph_hop_two(graph,type,device):
    """创建二跳子图"""
    if type == 'u':
        feature = graph.ndata['info']['u']
        graph_adj = dgl.edge_type_subgraph(graph,['src','tgt'])
        graph_combine = graph_adj['u', :, 'i'] # 合并边类型
        adj = graph_combine.adjacency_matrix().to_dense()
        # U*I * I*U
        graph_inverse_adj = dgl.edge_type_subgraph(graph,['src-by','tgt-by'])
        graph_combine = graph_inverse_adj['i', :, 'u']
        adj_inverse = graph_combine.adjacency_matrix().to_dense()
        neighbors_table = torch.matmul(adj,adj_inverse)
        src_start, src_end = dense_to_sparse(neighbors_table)
        graph_two = dgl.graph((src_start,src_end)).to(device) # # u.shape*u.shape
        graph_two.ndata['info'] = feature
    else:
        feature = graph.ndata['info']['i']
        graph_adj = dgl.edge_type_subgraph(graph, ['src', 'tgt'])
        graph_combine = graph_adj['u', :, 'i']
        adj = graph_combine.adjacency_matrix().to_dense()
        graph_inverse_adj = dgl.edge_type_subgraph(graph, ['src-by', 'tgt-by'])
        graph_combine = graph_inverse_adj['i', :, 'u']
        adj_inverse = graph_combine.adjacency_matrix().to_dense()
        neighbors_table = torch.matmul(adj_inverse,adj) # I.shape*I.shape
        src_start, src_end = dense_to_sparse(neighbors_table)
        graph_two = dgl.graph((src_start, src_end)).to(device)
        graph_two.ndata['info'] = feature

    return graph_two


def dense_to_sparse(matrix):
    """矩阵的互相转换"""
    idx = torch.nonzero(matrix).T  # 这里需要转置一下
    data = matrix[idx[0], idx[1]]
    coo_a = torch.sparse_coo_tensor(idx, data, matrix.shape)
    indices = coo_a._indices()
    # return indices
    return indices,data

def aggravate_domain(src,tgt,u_feature,device):
    graph_src = dgl.graph((src[0,:],src[1,:])).to(device)
    graph_tgt = dgl.graph((tgt[0,:],tgt[1,:])).to(device)
    graph_src.ndata['info'] = u_feature
    graph_tgt.ndata['info'] = u_feature
    net = GraphNet().to(device)
    u_feature_src = net(graph_src,u_feature)
    u_feature_tgt = net(graph_tgt,u_feature)
    return u_feature_src,u_feature_tgt



class NodeApplyModule(nn.Module):
    """node transfer"""
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}


class GCN(nn.Module):
    """graph agg,后续给它换了"""
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)
        self.gcn_msg = fn.copy_src(src='info', out='m')
        self.gcn_reduce = fn.sum(msg='m', out='info')

    def forward(self, graph, feature):
        graph.ndata['h'] = feature
        graph.update_all(self.gcn_msg, self.gcn_reduce)
        graph.apply_nodes(func=self.apply_mod)
        return graph.ndata.pop('h')


# class GCN_official(nn.Module):
#     def __init__(self, in_feats, h_feats, num_classes):
#         super(GCN, self).__init__()
#         self.conv1 = GraphConv(in_feats, h_feats)
#         self.conv2 = GraphConv(h_feats, num_classes)
#
#     def forward(self, g, in_feat):
#         x = F.relu(self.conv1(g, in_feat))
#         x = F.softmax(self.conv2(g, x))
#         return x

class NGCFLayer(nn.Module):
    """ref: https://github.com/dmlc/dgl/blob/master/examples/pytorch/NGCF/NGCF/model.py"""
    def __init__(self, in_size, out_size, norm_dict, dropout):
        super(NGCFLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        #weights for different types of messages
        self.W1 = nn.Linear(in_size, out_size, bias = True)
        self.W2 = nn.Linear(in_size, out_size, bias = True)
        self.W3 = nn.Linear(in_size, out_size, bias = True)

        #leaky relu
        self.leaky_relu = nn.LeakyReLU(0.2)

        #dropout layer
        self.dropout = nn.Dropout(dropout)

        #initialization
        torch.nn.init.xavier_uniform_(self.W1.weight)
        torch.nn.init.constant_(self.W1.bias, 0)
        torch.nn.init.xavier_uniform_(self.W2.weight)
        torch.nn.init.constant_(self.W2.bias, 0)

        #norm
        self.norm_dict = norm_dict

    def forward(self, g, feat_dict):
        funcs = {} #message and reduce functions dict
        #for each type of edges, compute messages and reduce them all
        for srctype, etype, dsttype in g.canonical_etypes:
            if srctype == dsttype: #for self loops
                messages = self.W1(feat_dict[srctype])
                g.nodes[srctype].data[etype] = messages   #store in ndata
                funcs[(srctype, etype, dsttype)] = (fn.copy_u(etype, 'm'), fn.sum('m', 'h'))  #define message and reduce functions
            else:
                src, dst = g.edges(etype=(srctype, etype, dsttype)) # sub_graph
                # print("A",src.shape)
                norm = self.norm_dict[(srctype, etype, dsttype)]

                # messages = norm * (self.W1(feat_dict[srctype][src]) + self.W1(
                #     feat_dict[srctype][src] * feat_dict[dsttype][dst]))  # compute messages
                messages = norm * (self.W1(feat_dict[srctype][src]) + self.W2(
                    feat_dict[srctype][src] * feat_dict[dsttype][dst]))  # compute messages，lenovo

                # if etype[:3]=="src": 
                #     messages = norm * (self.W1(feat_dict[srctype][src]) + self.W2(feat_dict[srctype][src]*feat_dict[dsttype][dst])) #compute messages
                # else:
                #     messages = norm * (self.W1(feat_dict[srctype][src]) + self.W3(feat_dict[srctype][src]*feat_dict[dsttype][dst])) #compute messages

                g.edges[(srctype, etype, dsttype)].data[etype] = messages  # store in edata
                funcs[(srctype, etype, dsttype)] = (fn.copy_e(etype, 'm'), fn.sum('m', 'h'))  #define message and reduce functions

        g.multi_update_all(funcs, 'sum') #update all, reduce by first type-wisely then across different types
        feature_dict={}
        for ntype in g.ntypes:
            h = self.leaky_relu(g.nodes[ntype].data['h']) #leaky relu
            h = self.dropout(h) #dropout
            h = F.normalize(h,dim=1,p=2) # l2 normalize
            feature_dict[ntype] = h
        return feature_dict



class GraphNet(nn.Module):
    def __init__(self):
        super(GraphNet, self).__init__()
        self.gcn1 = GCN(opt.KSIZE, opt.KSIZE, F.relu)
        # self.gcn2 = GCN(64, opt.KSIZE, F.relu)

    def forward(self, graph, feature):
        x = self.gcn1(graph, feature)
        # x = self.gcn2(graph, x)
        return x


class GraphNet_NGCF(nn.Module):
    """NGCF"""
    def __init__(self, g, in_size, layer_size, dropout, lmbd=1e-5):
        super(GraphNet_NGCF, self).__init__()
        self.lmbd = lmbd
        self.norm_dict = dict()
        for srctype, etype, dsttype in g.canonical_etypes:
            src, dst = g.edges(etype=(srctype, etype, dsttype))
            dst_degree = g.in_degrees(dst, etype=(srctype, etype, dsttype)).float() #obtain degrees
            src_degree = g.out_degrees(src, etype=(srctype, etype, dsttype)).float()
            norm = torch.pow(src_degree * dst_degree, -0.5).unsqueeze(1) #compute norm
            self.norm_dict[(srctype, etype, dsttype)] = norm

        self.layers = nn.ModuleList()
        self.layers.append(
            NGCFLayer(in_size, layer_size[0], self.norm_dict, dropout[0])
        )
        self.num_layers = len(layer_size)
        for i in range(self.num_layers-1):
            self.layers.append(
                NGCFLayer(layer_size[i], layer_size[i+1], self.norm_dict, dropout[i+1])
            )
        self.initializer = nn.init.xavier_uniform_

        #  embeddings for different types of nodes
        # self.feature_dict = nn.ParameterDict({
        #     ntype: nn.Parameter(self.initializer(torch.empty(g.num_nodes(ntype), in_size))) for ntype in g.ntypes
        # })
        self.feature_dict = nn.ParameterDict({
            ntype: nn.Parameter(g.nodes[ntype].data['info']) for ntype in g.ntypes
        })

    def forward(self, g, user_key, item_key):
        h_dict = {ntype : self.feature_dict[ntype] for ntype in g.ntypes}
        #obtain features of each layer and concatenate them all
        user_embeds = []
        item_embeds = []
        user_embeds.append(h_dict[user_key])
        item_embeds.append(h_dict[item_key])
        for layer in self.layers:
            h_dict = layer(g, h_dict)
            user_embeds.append(h_dict[user_key])
            item_embeds.append(h_dict[item_key])
        # user_embd = torch.cat(user_embeds, 1) # 3*H
        # item_embd = torch.cat(item_embeds, 1)
        user_embd = user_embeds[-1]
        item_embd = item_embeds[-1]
        return user_embd, item_embd


if __name__ == '__main__':
    # NGCF实现
    u = torch.tensor([0, 1, 1, 1, 2])
    i = torch.tensor([0, 0, 1, 2, 3])
    graph_data = {
        ('u', 'src', 'i'): (u, i),
        ('i', 'src-by', 'u'): (i, u)
    }  # i只是关系不同
    graph = dgl.heterograph(graph_data)
    model = GraphNet_NGCF(graph, 64, [64,64,64], [0.1,0.1,0.1], 0.1)
    model(graph, 'u', 'i', torch.tensor([0,1]).long(), torch.tensor([0,1]).long())


    # 创建整图和分场景子图
    # book_file ="/data/CrossRec/data/douban_book/ratings.dat"
    # movie_file = "/data/CrossRec/data/douban_movie/ratings.dat"
    # u_info = torch.randn(2718,3)
    # book_info = torch.randn(6777,3)
    # movie_info = torch.randn(9555,3) # 可以先传入列表筛选
    # src, tgt, item_src_map, item_tgt_map = read_graph(book_file,movie_file) # map这里用于对信息进行排序
    #
    # graph, _ = create_graph(src, tgt,u_info,book_info,movie_info,'cpu')
    #
    # sub_graph_src = dgl.edge_type_subgraph(graph, [('u', 'src', 'i'), ('i', 'src-by', 'u')])
    # sub_graph_tgt = dgl.edge_type_subgraph(graph, [('u', 'tgt', 'i'), ('i', 'tgt-by', 'u')])
    #
    # # 获取2跳邻居
    # neighbors_table_src, neighbors_table_tgt = neighbor_table(sub_graph_src,sub_graph_tgt)
    #
    # graph_two_hop = graph_hop_two(graph,type='u',device='cpu')
    #
    # gcn_msg = fn.copy_src(src='info', out='m') # 打包为message
    # gcn_reduce = fn.sum(msg='m', out='info') # 进行reduce
    # feature = graph_two_hop.ndata['info']
    #
    # net = GraphNet()
    # all_feature = net(graph_two_hop, feature)


    # 测试
    # u = torch.tensor([0,1,1,1,2])
    # i = torch.tensor([0,0,1,2,3])
    # graph_data = {
    #     ('u', 'src', 'i'): (u, i),
    #     ('i', 'src-by', 'u'): (i, u)
    # }  # i只是关系不同
    # graph = dgl.heterograph(graph_data)
    # adj = graph.adjacency_matrix(etype='src').to_dense()
    # adj_inverse = graph.adjacency_matrix(etype='src-by').to_dense()
    # neighbor = torch.matmul(adj,adj_inverse)
    # all_fea = (neighbor.t() / torch.norm(neighbor, p=2, dim=1)).t()
    # print(neighbor)
    # print(all_fea)






