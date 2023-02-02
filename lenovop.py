#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/3 19:25
# @Author  : Jack Zhao
# @Site    : 
# @File    : lenovop.py
# @Software: PyCharm

# #Desc: 

import pandas as pd
import numpy as np
import nltk

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from stanfordcorenlp import StanfordCoreNLP
from doubanp import format_str
from utils import print_info
from collections import defaultdict

root_p = "/data/CrossRec/data/"

#get chinese nlp
# nlp = StanfordCoreNLP('/miniconda/miniconda/lib/python3.8/site-packages/stanfordcorenlp/stanford-corenlp-full-2022-07-25',lang='zh')
# clean stops
# nltk.download('stopwords')


def user_unify_map():
    # 这里user要统一编码
    # shop
    df_r_shop = pd.read_csv("/data/CrossRec/data/lenovo/click1.csv", header=None)
    df_r_shop.columns = ["u", "i", "score", "time"]
    df_r_shop = df_r_shop[df_r_shop["score"] == 1][['u', 'i', 'score']]
    df_r_shop = df_r_shop.groupby("u").filter(lambda x: (len(x) >= 10))  # 用的是10次

    # community
    df_r_com = pd.read_csv("/data/CrossRec/data/lenovo/click2.csv", header=None)
    df_r_com.columns = ["u", "i", "score", "time"]
    df_r_com = df_r_com[df_r_com["score"] == 1][['u', 'i', 'score']]
    df_r_com = df_r_com.groupby("u").filter(lambda x: (len(x) >= 5))  # 用的是10

    # 统一编码
    df_r = pd.concat((df_r_shop,df_r_com),axis=0)
    print("overall user num", len(df_r['u'].unique()))
    u_unique = df_r['u'].unique()
    # 要让overlap的id排在最后
    df_overlap = pd.merge(df_r_shop,df_r_com,on='u',how='inner')
    df_r = df_r[~df_r.u.isin(df_overlap['u'].unique().tolist())]
    print("overlap user num",len(df_overlap['u'].unique()),len(df_r['u'].unique()))
    # u_map = dict(zip(u_unique, list(range(len(df_r['u'].unique())+len(df_overlap['u'].unique())))))
    u_map_nonoverlap = dict(zip(df_r['u'].unique(), list(range(1,1+len(df_r['u'].unique())))))
    u_map_overlap = dict(zip(df_overlap['u'].unique(), list(range(len(df_r['u'].unique())+1,1+len(df_r['u'].unique())+len(df_overlap['u'].unique())))))
    u_map_nonoverlap.update(u_map_overlap)
    print("map num",len(u_map_nonoverlap))
    return u_map_nonoverlap,u_unique

u_map,u_unique_lis = user_unify_map()


def search_entity(row,entity_df):
    """如果没有则传入无"""
    if row['u'] in entity_df['u']:
        return entity_df[row['u']]
    else:
        return "无"



def merge(row):
    """entity处理"""
    str_cleaned = ""
    if row["entity"] != "None":
        str_cleaned = format_str(row["entity"], 1)
    if str_cleaned=='':
       return ""
    else:
        str_cleaned = str_cleaned[:20000] # 太长会decode失败
        words = nlp.word_tokenize(str_cleaned)
        return words



def doc_emb_cop(u_file, i_file, vec_num, domain="shop"):
    """这里使用user/item合计"""
    print("Start!")
    documents = u_file[domain +'_info'].values.tolist() + \
                i_file[domain +'_info'].values.tolist()
    frequency = defaultdict(int)
    for text in documents:
        for token in text:
            frequency[token] += 1
    # 大于1的数据
    texts = [[token for token in text if frequency[token] > 1]
             for text in documents]
    # train the model
    documents = [TaggedDocument(doc, [int(i)]) for i, doc in enumerate(documents)]
    # documents: Users + Movies
    docs = documents
    model = Doc2Vec(docs, vector_size=vec_num, window=2, min_count=5, negative=5, workers=6)
    model.train(docs, total_examples=model.corpus_count, epochs=50)
    model.save(root_p + "lenovo_" + domain + "/Doc2vec_lenovo_%s_VSize%02d.model" % (domain,vec_num))
    vectors = model.docvecs.vectors_docs
    print("End!")



def doc_emb(i_file, vec_num, flag, domain="community"):
    """这里进行处理"""
    print("Start!")
    documents = i_file[domain +'_info'].values.tolist()

    frequency = defaultdict(int)
    for text in documents:
        for token in text:
            frequency[token] += 1

    # 大于1的数据
    texts = [[token for token in text if frequency[token] > 1]
             for text in documents]

    # train the model
    documents = [TaggedDocument(doc, [int(i)]) for i, doc in enumerate(documents)]
    # documents: Users + Movies
    docs = documents
    model = Doc2Vec(docs, vector_size=vec_num, window=2, min_count=5, negative=5, workers=6)
    model.train(docs, total_examples=model.corpus_count, epochs=50)
    model.save(root_p + "lenovo_" + domain + "/Doc2vec_lenovo_%s_%s_VSize%02d.model" % (domain,flag,vec_num))
    # vectors = model.docvecs.vectors_docs
    print("End!")


def shop(vec_num):
    """处理商城数据"""
    print("Start!")

    # 用户属性
    df_u = pd.read_csv("/data/CrossRec/data/lenovo/u1_entity.csv")
    df_u["shop_info"] = df_u.apply(merge, axis=1)

    # 商品属性
    df_i = pd.read_csv("/data/CrossRec/data/lenovo/item_1_entity.csv",encoding="utf-8")
    id_feature = ["goods_business_cd","product_category_name","product_type_name"]
    num_feature = ["pc_price"]
    # cat_feature = ["product_category_name","product_type_name"]
    text_feature = ["entity"]
    for i in num_feature:
        df_i[[i]] = df_i[[i]].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

    # for i in cat_feature: # 这里使用多hot编码
    #     df_i[i] = df_i[i].astype('category')
    #     df_i[i] = df_i[i].cat.codes
    #     df_i = pd.concat((df_i, pd.get_dummies(df_i[i], prefix=i)), axis=1)

    # 交互
    df_r = pd.read_csv("/data/CrossRec/data/lenovo/click1.csv",header=None)
    df_r.columns = ["u","i","score","time"]
    df_r = df_r[df_r["score"]==1][['u','i','score']]
    df_r = df_r.groupby("u").filter(lambda x: (len(x) >= 10))  # 用的是10次

    print("u",len(df_r['u'].unique()),df_r['u'].max(),df_r['u'].min())
    print("i",len(df_r['i'].unique()),df_r['i'].max(),df_r['i'].min())
    # 重新编码
    # u_map = dict(zip(df_r['u'].unique(),list(range(len(df_r['u'].unique())))))
    # u_unique_lis = df_r['u'].unique().tolist()
    df_r['u'] = df_r['u'].map(u_map)
    i_map = dict(zip(df_r['i'].unique(),list(range(1,1+len(df_r['i'].unique())))))
    i_unique_lis = df_r['i'].unique().tolist()
    df_r['i'] = df_r['i'].map(i_map)
    df_r.to_csv("/data/CrossRec/data/lenovo_shop/ratings_p.csv")

    # i实体补充
    df = pd.DataFrame({"goods_business_cd": i_unique_lis})
    df_join = pd.merge(df,df_i,on='goods_business_cd',how='left')
    df_join["goods_business_cd"] = df_join['goods_business_cd'].map(i_map)
    df_join.sort_values(by="goods_business_cd", inplace=True, ascending=True)
    df_join['entity'] = df_join['entity'].fillna("无")
    df_join["shop_info"] = df_join.apply(merge, axis=1)
    # doc_emb(df_join, vec_num, flag='i',domain="shop") # 实体特征单独存储
    # 这里和Douban数据集不一样，这里并没有使用user和item的交互来补充其特征
    df = pd.DataFrame({"lenovoid":u_unique_lis})
    df_join_u = pd.merge(df,df_u,on='lenovoid',how='left')[['lenovoid','entity']]
    df_join_u["lenovoid"] = df_join_u['lenovoid'].map(u_map)
    df_join_u['entity'] = df_join_u['entity'].fillna("无")
    df_join_u.sort_values(by="lenovoid", inplace=True, ascending=True)
    df_join_u["shop_info"] = df_join_u.apply(merge, axis=1)
    # doc_emb(df_join_u, vec_num, flag='u', domain="shop") # 实体特征单独存储

    doc_emb_cop(df_join_u,df_join, vec_num, domain="shop")






def community(vec_num):
    """处理社区数据"""
    print("Start!")
    # 用户属性
    # 用户属性
    df_u = pd.read_csv("/data/CrossRec/data/lenovo/u2_entity.csv")
    df_u["community_info"] = df_u.apply(merge, axis=1)

    # 商品属性
    df_i = pd.read_csv("/data/CrossRec/data/lenovo/item_2_entity.csv",index_col=0)
    id_feature = ["tid", "fid", "sortid", "typeid","authorid","highlight"]
    num_feature = ["views","favtimes","comments","displayorder"]
    cat_feature = ["digest"]
    text_feature = ["entity"]
    for i in num_feature:
        df_i[[i]] = df_i[[i]].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

    for i in cat_feature:
        df_i[i] = df_i[i].astype('category')
        df_i[i] = df_i[i].cat.codes
        df_i = pd.concat((df_i,pd.get_dummies(df_i[i],prefix=i)),axis=1)

    # 点击
    df_r = pd.read_csv("/data/CrossRec/data/lenovo/click2.csv", header=None)
    df_r.columns = ["u", "i", "score", "time"]
    df_r = df_r[df_r["score"] == 1][['u', 'i', 'score']]
    df_r = df_r.groupby("u").filter(lambda x: (len(x) >= 5))  # 用的是5次
    print("u", len(df_r['u'].unique()), df_r['u'].max(), df_r['u'].min())
    print("i", len(df_r['i'].unique()), df_r['i'].max(), df_r['i'].min())
    # 重新编码
    # u_map = dict(zip(df_r['u'].unique(), list(range(len(df_r['u'].unique())))))
    # u_unique_lis = df_r['u'].unique().tolist()
    df_r['u'] = df_r['u'].map(u_map)
    i_map = dict(zip(df_r['i'].unique(), list(range(1,1+len(df_r['i'].unique())))))
    i_unique_lis = df_r['i'].unique().tolist()
    df_r['i'] = df_r['i'].map(i_map)
    df_r.to_csv("/data/CrossRec/data/lenovo_community/ratings_p.csv")

    # 特征
    df = pd.DataFrame({"tid": i_unique_lis})
    df_join = pd.merge(df,df_i,on='tid',how='left')
    df_join["tid"] = df_join['tid'].map(i_map)
    df_join.sort_values(by="tid", inplace=True, ascending=True)
    df_join['entity'] = df_join['entity'].fillna("无")
    df_join["community_info"] = df_join.apply(merge,axis=1)
    # doc_emb(df_join, vec_num, flag='i',domain="community") # 实体特征单独存储

    df = pd.DataFrame({"lenovoid":u_unique_lis})
    df_join_u = pd.merge(df,df_u,on='lenovoid',how='left')[['lenovoid','entity']]
    df_join_u["lenovoid"] = df_join_u['lenovoid'].map(u_map)
    df_join_u['entity'] = df_join_u['entity'].fillna("无")
    df_join_u.sort_values(by="lenovoid", inplace=True, ascending=True)
    df_join_u["community_info"] = df_join_u.apply(merge,axis=1)
    # doc_emb(df_join_u, vec_num, flag='u', domain="community") # 实体特征单独存储
    doc_emb_cop(df_join_u,df_join, vec_num, domain="community")





if __name__ == '__main__':
    # print("XXXXXXXXXXX")


    # density
    df_r = pd.read_csv("/data/CrossRec/data/lenovo/click1.csv", header=None)
    df_r.columns = ["u", "i", "score", "time"]
    df_r = df_r[df_r["score"] == 1][['u', 'i', 'score']]

    u = len(df_r['u'].unique())
    u_o = set(df_r['u'].unique().tolist())
    i = len(df_r['i'].unique())
    inter = df_r.shape[0]
    print("u:{},i:{},inter:{},density:{}".format(u,i,inter,round(inter/(u*i),6)))

    df_r = pd.read_csv("/data/CrossRec/data/lenovo/click2.csv", header=None)
    df_r.columns = ["u", "i", "score", "time"]
    df_r = df_r[df_r["score"] == 1][['u', 'i', 'score']]

    u = len(df_r['u'].unique())
    i = len(df_r['i'].unique())
    u_o_2 = set(df_r['u'].unique().tolist())

    inter = df_r.shape[0]
    print("u:{},i:{},inter:{},density:{}".format(u,i,inter,round(inter/(u*i),6)))

    print('overlap num: ',len(u_o & u_o_2))




