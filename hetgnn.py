#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/3 16:11
# @Author  : Jack Zhao
# @Site    : 
# @File    : hetgnn.py
# @Software: PyCharm

# #Desc:

import dgl
import torch as th
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F


from abc import ABCMeta
import torch.nn as nn
import argparse


class BaseModel(nn.Module, metaclass=ABCMeta):
    @classmethod
    def build_model_from_args(cls, args, hg):
        r"""
        Build the model instance from args and hg.
        So every subclass inheriting it should override the method.
        """
        raise NotImplementedError("Models must implement the build_model_from_args method")

    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, *args):
        r"""
        The model plays a role of encoder. So the forward will encoder original features into new features.
        Parameters
        -----------
        hg : dgl.DGlHeteroGraph
            the heterogeneous graph
        h_dict : dict[str, th.Tensor]
            the dict of heterogeneous feature
        Return
        -------
        out_dic : dict[str, th.Tensor]
            A dict of encoded feature. In general, it should ouput all nodes embedding.
            It is allowed that just output the embedding of target nodes which are participated in loss calculation.
        """
        raise NotImplementedError

    def extra_loss(self):
        r"""
        Some model want to use L2Norm which is not applied all parameters.
        Returns
        -------
        th.Tensor
        """
        raise NotImplementedError

    def h2dict(self, h, hdict):
        pre = 0
        out_dict = {}
        for i, value in hdict.items():
            out_dict[i] = h[pre:value.shape[0]+pre]
            pre += value.shape[0]
        return out_dict

    def get_emb(self):
        r"""
        Return the embedding of a model for further analysis.
        Returns
        -------
        numpy.array
        """
        raise NotImplementedError





class HetGNN(BaseModel):
    r"""
    HetGNN[KDD2019]-
    `Heterogeneous Graph Neural Network <https://dl.acm.org/doi/abs/10.1145/3292500.3330961>`_
    `Source Code Link <https://github.com/chuxuzhang/KDD2019_HetGNN>`_

    The author of the paper only gives the academic dataset.

    Attributes
    -----------
    Het_Aggrate : nn.Module
        Het_Aggregate
    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(hg, args)


    def __init__(self, hg, args):
        super(HetGNN, self).__init__()
        print(hg)
        self.Het_Aggregate = Het_Aggregate(hg.ntypes, args.dim)
        self.ntypes = hg.ntypes
        self.device = args.device

        self.loss_fn = HetGNN.compute_loss

    def forward(self, hg, h=None):
        if h is None:
            h = self.extract_feature(hg, self.ntypes)
        x = self.Het_Aggregate(hg, h)
        return x


    def evaluator(self):
        self.link_preddiction()
        self.node_classification()

    def get_embedding(self):
        input_features = self.model.extract_feature(self.hg, self.hg.ntypes)
        x = self.model(self.model.preprocess(self.hg, self.args).to(self.args.device), input_features)
        return x

    def link_preddiction(self):
        x = self.get_embedding()
        self.model.lp_evaluator(x[self.category].to('cpu').detach(), self.train_batch, self.test_batch)

    def node_classification(self):
        x = self.get_embedding()
        self.model.nc_evaluator(x[self.category].to('cpu').detach(), self.labels, self.train_idx, self.test_idx)

    @staticmethod
    def compute_loss(pos_score, neg_score):
        # an example hinge loss
        loss = []
        for i in pos_score:
            loss.append(F.logsigmoid(pos_score[i]))
            loss.append(F.logsigmoid(-neg_score[i]))
        loss = th.cat(loss)
        return -loss.mean()

    @staticmethod
    def extract_feature(g, ntypes):
        input_features = {}
        for n in ntypes:
            ndata = g.srcnodes[n].data
            data = {}
            data['dw_embedding'] = ndata['dw_embedding']
            data['abstract'] = ndata['abstract']
            if n == 'paper':
                data['title'] = ndata['title']
                data['venue'] = ndata['venue']
                data['author'] = ndata['author']
                data['reference'] = ndata['reference']
            input_features[n] = data

        return input_features

    @staticmethod
    def pred(edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            return edge_subgraph.edata['score']



class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            return edge_subgraph.edata['score']


class Het_Aggregate(nn.Module):
    r"""
    The whole model of HetGNN

    Attributes
    -----------
    content_rnn : nn.Module
        het_content_encoder
    neigh_rnn : nn.Module
        aggregate_het_neigh
    atten_w : nn.ModuleDict[str, nn.Module]


    """
    def __init__(self, ntypes, dim):
        super(Het_Aggregate, self).__init__()
        # ntypes means nodes type name
        self.ntypes =ntypes
        self.dim = dim

        self.content_rnn = het_content_encoder(dim)
        self.neigh_rnn = aggregate_het_neigh(ntypes, dim)

        self.atten_w = nn.ModuleDict({})
        for n in self.ntypes:
            self.atten_w[n] = nn.Linear(in_features=dim * 2, out_features=1)

        self.softmax = nn.Softmax(dim=1)
        self.activation = nn.LeakyReLU()
        self.drop = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(dim)
        self.embed_d = dim

    def forward(self, hg, h_dict):
        with hg.local_scope():
            content_h = {}
            for ntype, h in h_dict.items():
                content_h[ntype] = self.content_rnn(h)

            neigh_h = self.neigh_rnn(hg, content_h)
            # the content feature of the dst nodes
            dst_h = {k: v[:hg.number_of_dst_nodes(k)] for k, v in content_h.items()}
            out_h = {}
            for n in self.ntypes:
                d_h = dst_h[n]
                batch_size = d_h.shape[0]
                concat_h = []
                concat_emd = []
                for i in range(len(neigh_h[n])):
                    concat_h.append(th.cat((d_h, neigh_h[n][i]), 1))
                    concat_emd.append(neigh_h[n][i])
                concat_h.append(th.cat((d_h, d_h), 1))
                concat_emd.append(d_h)

                concat_h = th.hstack(concat_h).view(batch_size * (len(self.ntypes) + 1), self.dim *2)
                atten_w = self.activation(self.atten_w[n](concat_h)).view(batch_size, len(self.ntypes) + 1)

                atten_w = self.softmax(atten_w).view(batch_size, 1, 4)

                # weighted combination
                concat_emd = th.hstack(concat_emd).view(batch_size, len(self.ntypes) + 1, self.dim)

                weight_agg_batch = th.bmm(atten_w, concat_emd).view(batch_size, self.dim)
                out_h[n] = weight_agg_batch
            return out_h


class het_content_encoder(nn.Module):
    r"""
    The Encoding Heterogeneous Contents(C2) in the paper
    For a specific node type, encoder different content features with a LSTM.

    In paper, it is (b) NN-1: node heterogeneous contents encoder in figure 2.

    Parameters
    ------------
    dim : int
        input dimension

    Attributes
    ------------
    content_rnn : nn.Module
        nn.LSTM encode different content feature
    """
    def __init__(self, dim):
        super(het_content_encoder, self).__init__()
        self.content_rnn = nn.LSTM(dim, int(dim / 2), 1, batch_first=True, bidirectional=True)
        self.content_rnn.flatten_parameters()
        self.dim = dim

    def forward(self, h_dict):
        r"""

        Parameters
        ----------
        h_dict: dict[str, th.Tensor]
            key means different content feature

        Returns
        -------
        content_h : th.tensor
        """
        concate_embed = []
        for _, h in h_dict.items():
            concate_embed.append(h)
        concate_embed = th.cat(concate_embed, 1)
        print(concate_embed.shape)
        concate_embed = concate_embed.view(concate_embed.shape[0], -1, self.dim)
        all_state, last_state = self.content_rnn(concate_embed)
        out_h = th.mean(all_state, 1).squeeze()
        return out_h


class aggregate_het_neigh(nn.Module):
    r"""
    It is a Aggregating Heterogeneous Neighbors(C3)
    Same Type Neighbors Aggregation

    """
    def __init__(self, ntypes, dim):
        super(aggregate_het_neigh, self).__init__()
        self.neigh_rnn = nn.ModuleDict({})
        self.ntypes =ntypes
        for n in ntypes:
            self.neigh_rnn[n] = lstm_aggr(dim)

    def forward(self, hg, inputs):
        with hg.local_scope():
            outputs = {}
            for i in self.ntypes:
                outputs[i] = []
            if isinstance(inputs, tuple) or hg.is_block:
                if isinstance(inputs, tuple):
                    src_inputs, dst_inputs = inputs
                else:
                    src_inputs = inputs
                    dst_inputs = {k: v[:hg.number_of_dst_nodes(k)] for k, v in inputs.items()}

                for stype, etype, dtype in hg.canonical_etypes:
                    rel_graph = hg[stype, etype, dtype]
                    if rel_graph.number_of_edges() == 0:
                        continue
                    if stype not in src_inputs or dtype not in dst_inputs:
                        continue
                    dstdata = self.neigh_rnn[stype](
                        rel_graph,
                        (src_inputs[stype], dst_inputs[dtype]))
                    outputs[dtype].append(dstdata)
            else:
                for stype, etype, dtype in hg.canonical_etypes:
                    rel_graph = hg[stype, etype, dtype]
                    if rel_graph.number_of_edges() == 0:
                        continue
                    if stype not in inputs:
                        continue
                    dstdata = self.neigh_rnn[stype](
                        rel_graph,
                        inputs[stype])
                    outputs[dtype].append(dstdata)
            return outputs


class lstm_aggr(nn.Module):
    r"""
    Aggregate the same neighbors with LSTM
    """

    def __init__(self, dim):
        super(lstm_aggr, self).__init__()
        self.lstm = nn.LSTM(dim, int(dim / 2), 1, batch_first=True, bidirectional=True)

        self.lstm.flatten_parameters()
    def _lstm_reducer(self, nodes):
        m = nodes.mailbox['m']  # (B, L, D)
        batch_size = m.shape[0]
        all_state, last_state = self.lstm(m)
        return {'neigh': th.mean(all_state, 1)}

    def forward(self, g, inputs):
        with g.local_scope():
            if isinstance(inputs, tuple) or g.is_block:
                if isinstance(inputs, tuple):
                    src_inputs, dst_inputs = inputs
                else:
                    src_inputs = inputs
                    dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
                g.srcdata['h'] = src_inputs
                g.update_all(fn.copy_u('h', 'm'), self._lstm_reducer)
                h_neigh = g.dstdata['neigh']
            else:
                g.srcdata['h'] = inputs
                g.update_all(fn.copy_u('h', 'm'), self._lstm_reducer)
                h_neigh = g.dstdata['neigh']
            return h_neigh



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test')
    # type是要传入的参数的数据类型  help是该参数的提示信息
    parser.add_argument('--device', type=str, default="cpu",help='device')
    parser.add_argument('--dim', type=int, default=9, help='emb dim')

    args = parser.parse_args()

    # 获得传入的参数
    u = th.tensor([0, 1, 1, 1, 2])
    i = th.tensor([0, 0, 1, 2, 3])
    graph_data = {
        ('u', 'src', 'i'): (u, i),
        ('i', 'src-by', 'u'): (i, u)
    }  # i只是关系不同
    graph = dgl.heterograph(graph_data)
    u_feature = th.randn(3,2,3)
    u_feature = dict(zip(list(range(u_feature.shape[0])), u_feature))
    i_feature = th.randn(4,2,3)
    i_feature = dict(zip(list(range(i_feature.shape[0])), i_feature))
    het = HetGNN(graph,args)
    het(graph,{"u": u_feature,"i":i_feature})


