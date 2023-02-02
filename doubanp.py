#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/26 10:31
# @Author  : Jack Zhao
# @Site    :
# @File    : doubanp.py
# @Software: PyCharm

# #Desc: 数据预处理
import pandas as pd
import nltk
import datetime
import numpy as np

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from stanfordcorenlp import StanfordCoreNLP
from utils import print_info
from collections import defaultdict

root = "/data/CrossRec/data/douban_feature_raw/"
root_to = "/data/CrossRec/data/douban_feature/"
root_p = "/data/CrossRec/data/"
#get chinese nlp
nlp = StanfordCoreNLP('/miniconda/miniconda/lib/python3.8/site-packages/stanfordcorenlp/stanford-corenlp-full-2022-07-25',lang='zh') # unzip-o,改名,jar，改名
# clean stops
nltk.download('stopwords')


def is_chinese(uchar):
    """is this a chinese word?"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def is_number(uchar):
    """is this unicode a number?"""
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False


def is_alphabet(uchar):
    """is this unicode an English word?"""
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False


def format_str(content, lag):
    content_str = ''
    if lag == 0:  # English
        for i in content:
            if is_alphabet(i):
                content_str = content_str + i
    if lag == 1:  # Chinese
        for i in content:
            if is_chinese(i):
                content_str = content_str + i
    if lag == 2:  # Number
        for i in content:
            if is_number(i):
                content_str = content_str + i
    return content_str


def merge(row):
    """user"""
    str_cleaned = ""
    if row["living_place"]!="None":
        str_cleaned = format_str(row["living_place"], 1)
    if row["self_statement"]!="None":
        str_cleaned = str_cleaned + format_str(row["self_statement"], 1)
    if str_cleaned=='':
       return ""
    else:
        words = nlp.word_tokenize(str_cleaned)
        return words


def merge_movie(row):
    """movie"""
    str_cleaned = ""
    if row["name"] !="None":
        str_cleaned += format_str(row["name"], 1)
    if row["director"] !="None":
        str_cleaned += format_str(row["director"], 1)
    if row["summary"] !="None":
        str_cleaned += format_str(row["summary"], 1)
    if row["writer"] !="None":
        str_cleaned += format_str(row["writer"], 1)
    if row["country"] !="None":
        str_cleaned += format_str(row["country"], 1)
    if row["language"] is not None:
        str_cleaned += format_str(row["language"], 1)
    if row["tag"] is not None:
        str_cleaned += format_str(row[11], 1)
    if str_cleaned == '':
        return ""
    else:
        words = nlp.word_tokenize(str_cleaned)
        return words


def merge_review(row):
    """user->movie/book/music review"""
    str_cleaned = ''
    if row["comment"] !="None":
        str_cleaned += format_str(row["comment"], 1)
    if row["labels"] !="None":
        str_cleaned += format_str(row["labels"], 1)
    if str_cleaned == '':
        return ""
    else:
        words = nlp.word_tokenize(str_cleaned)
        return words



@print_info
def movie_domain():
    # file_u记录着用户的profile
    print("Start!")
    file_u = read_file(root+"users_cleaned.txt")
    file_u = file_u.fillna(value="None")
    file_u["user_info"]=file_u.apply(merge,axis=1)
    print("Load User Profiles: Finished!")
    print("user num: ",file_u.shape[0])

    # file_m记录每个movie的profile
    file_m = read_file(root+"movies_cleaned.txt") # 最后一列是重新编号的id
    file_m = file_m.fillna(value="None")
    file_m["movie_info"] = file_m.apply(merge_movie, axis=1)
    print("movie num: ",file_m.shape[0])
    print("Load Moive Details (DoubanMovie): Finished!")

    # file_inter记录交互
    file_mr = read_file(root+"moviereviews_cleaned.txt")
    file_mr = file_mr.groupby("movie_id").filter(lambda x: (len(x) > 20))  # 用的是20次
    file_inter = file_mr[["user_id", "movie_id", "rating"]]
    file_inter.to_csv(root_to+"um_inter.csv") # 没有上一条老出现多行的问题

    movie_list = file_mr['movie_id'].unique().tolist()
    file_mr = file_mr.fillna(value="None")
    file_mr["common"] = file_mr.apply(merge_review, axis=1) # 这块内容会被组合
    print("Inter num: ", file_mr.shape[0])
    print("Load User Reviews (DoubanMoive): Finished!")

    file_gu = file_mr.groupby(['user_id'])['common'].apply(list).to_frame().reset_index(drop=False)
    file_gm = file_mr.groupby(['movie_id'])['common'].apply(list).to_frame().reset_index(drop=False)

    for i in range(file_u.shape[0]):
        id = file_u['UID'].iloc[i] # 重编码了
        if type(file_u['user_info'][i]) != str:
            file_u['user_info'][i].extend(file_gu[file_gu['user_id']==id]['common'].values.tolist())
        else:
            file_u['user_info'][i] = [file_u['user_info'][i]] + file_gu[file_gu['user_id'] == id]['common'].values.tolist()

    file_u.to_csv(root_to+"um_feature.csv")

    for i in range(file_m.shape[0]):
        id = file_m['UID'].iloc[i]
        if type(file_m['movie_info'][i]) != str:
            file_m['movie_info'][i].extend(file_gm[file_gm['movie_id'] == id]['common'].values.tolist())
        else:
            file_m['movie_info'][i] = [file_m['movie_info'][i]] + file_gm[file_gm['movie_id'] == id]['common'].values.tolist()

    file_m = file_m[file_m.UID.isin(movie_list)]

    file_m.to_csv(root_to+"m_feature.csv")


@print_info
def book_domain():
    print("Start!")

    file_u = read_file(root + "users_cleaned.txt")
    file_u = file_u.fillna(value="None")
    file_u["user_info"] = file_u.apply(merge, axis=1)
    print("Load User Profiles: Finished!")
    print("user num: ", file_u.shape[0])

    file_b = read_file(root + "books_cleaned.txt")  # 最后一列是重新编号的id
    file_b['book_info'] = ""
    print("book num: ", file_b.shape[0])
    print("Load Book Details (DoubanBook): Finished!")


    file_br = read_file(root + "bookreviews_cleaned.txt")
    file_br = file_br.groupby("book_id").filter(lambda x: (len(x) > 5))  # 用的是5次
    file_inter = file_br[["user_id", "book_id", "rating"]]
    file_inter.to_csv(root_to+"ub_inter.csv")
    book_list = file_br['book_id'].unique().tolist()

    file_br = file_br.fillna(value="None")
    file_br["common"] = file_br.apply(merge_review, axis=1)
    print("Load User Reviews: Finished!")
    print("inter num: ", file_br.shape[0])

    file_gu = file_br.groupby(['user_id'])['common'].apply(list).to_frame().reset_index(drop=False)
    file_gb = file_br.groupby(['book_id'])['common'].apply(list).to_frame().reset_index(drop=False)


    for i in range(file_u.shape[0]):
        id = file_u['UID'].iloc[i]  # 重编码了
        if type(file_u['user_info'][i]) != str:
            file_u['user_info'][i].extend(file_gu[file_gu['user_id'] == id]['common'].values.tolist())
        else:
            file_u['user_info'][i] = [file_u['user_info'][i]] + file_gu[file_gu['user_id'] == id]['common'].values.tolist()

    file_u.to_csv(root_to + "ub_feature.csv")


    for i in range(file_b.shape[0]):
        id = file_b['UID'].iloc[i]
        if type(file_b['book_info'][i]) != str:
            file_b['book_info'][i].extend(file_gb[file_gb['book_id'] == id]['common'])
        else:
            file_b['book_info'][i] = [file_b['book_info'][i]] + file_gb[file_gb['book_id'] == id]['common'].values.tolist()
    file_b = file_b[file_b.UID.isin(book_list)]

    file_b.to_csv(root_to + "b_feature.csv")


@print_info
def music_domain():
    print("Start!")
    file_u = read_file(root + "users_cleaned.txt")
    file_u = file_u.fillna(value="None")
    file_u["user_info"] = file_u.apply(merge, axis=1)
    print("Load User Profiles: Finished!")
    print("user num: ", file_u.shape[0])

    file_mu = read_file(root + "music_cleaned.txt")  # 最后一列是重新编号的id
    file_mu['music_info'] = ""
    print("music num: ", file_mu.shape[0])
    print("Load Music Details (DoubanMusic): Finished!")

    file_mur = read_file(root + "musicreviews_cleaned.txt")
    file_mur = file_mur.groupby("music_id").filter(lambda x: (len(x) > 5))  # 用的是5次
    file_inter = file_mur[["user_id", "music_id", "rating"]]

    file_inter.to_csv(root_to+"umu_inter.csv")

    music_list = file_mur['music_id'].unique().tolist()

    file_mur = file_mur.fillna(value="None")
    file_mur["common"] = file_mur.apply(merge_review, axis=1)
    print("Load User Reviews: Finished!")
    print("inter num: ", file_mur.shape[0])

    file_gu = file_mur.groupby(['user_id'])['common'].apply(list).to_frame().reset_index(drop=False)
    file_gmu = file_mur.groupby(['music_id'])['common'].apply(list).to_frame().reset_index(drop=False)

    for i in range(file_u.shape[0]):
        id = file_u['UID'].iloc[i]  # 重编码了
        if type(file_u['user_info'][i]) !=str:
            file_u['user_info'][i].extend(file_gu[file_gu['user_id'] == id]['common'].values.tolist())
        else:
            file_u['user_info'][i] = [file_u['user_info'][i]] + file_gu[file_gu['user_id'] == id]['common'].values.tolist()

    file_u.to_csv(root_to + "umu_feature.csv")

    for i in range(file_mu.shape[0]):
        id = file_mu['UID'].iloc[i]
        if type(file_mu['music_info'][i]) != str:
            file_mu['music_info'][i].extend(file_gmu[file_gmu['music_id'] == id]['common'].values.tolist())
        else:
            file_mu['music_info'][i] = [file_mu['music_info'][i]] + file_gmu[file_gmu['music_id'] == id]['common'].values.tolist()


    file_mu = file_mu[file_mu.UID.isin(music_list)]
    file_mu.to_csv(root_to + "mu_feature.csv")


def re_id(rating_file,i_file,u_file,domain,vector_nums):
    """对music/movie/music的id进行重新编码"""
    print("开始处理{}的数据!".format(domain))
    if domain == "movie":
        id_unique = rating_file['movie_id'].unique().tolist()
        re_id_movie = dict(zip(id_unique, list(range(1,len(id_unique)+1))))
        i_file['UID'] = i_file['UID'].map(re_id_movie)
        rating_file['movie_id'] = rating_file['movie_id'].map(re_id_movie)

        u_file = u_file[["UID","user_info"]]
        i_file = i_file[["UID","movie_info"]]
        rating_file = rating_file[["user_id","movie_id","rating"]]
        u_file['user_info'] = u_file['user_info'].apply(lambda x: eval(x))
        u_file['user_info'] = u_file['user_info'].apply(lambda x: eval('[%s]' % repr(x).replace('[', '').replace(']', '')))
        i_file["movie_info"] = i_file['movie_info'].apply(lambda x: eval(x))
        i_file['movie_info'] = i_file['movie_info'].apply(lambda x: eval('[%s]' % repr(x).replace('[', '').replace(']', '')))
        for vector_num in vector_nums:
            doc_emb(u_file, i_file, vector_num, domain="movie")
        rating_file.to_csv("/data/CrossRec/data/douban_movie/ratings_p.csv")
        i_file.to_csv("/data/CrossRec/data/douban_movie/movie_feature_p.csv")
        u_file.to_csv("/data/CrossRec/data/douban_movie/u_feature_p.csv")
    elif domain == "book":
        id_unique = rating_file['book_id'].unique().tolist()
        re_id_book = dict(zip(id_unique,list(range(1,len(id_unique)+1))))
        i_file['UID'] = i_file['UID'].map(re_id_book)
        rating_file['book_id'] = rating_file['book_id'].map(re_id_book)
        u_file = u_file[["UID", "user_info"]]
        i_file = i_file[["UID", "book_info"]]
        rating_file = rating_file[["user_id", "book_id","rating"]]
        u_file['user_info'] = u_file['user_info'].apply(lambda x: eval(x))
        u_file['user_info'] = u_file['user_info'].apply(lambda x: eval('[%s]' % repr(x).replace('[', '').replace(']', '')))
        i_file["book_info"] = i_file['book_info'].apply(lambda x: eval(x))
        i_file['book_info'] = i_file['book_info'].apply(lambda x: eval('[%s]' % repr(x).replace('[', '').replace(']', '')))
        for vector_num in vector_nums:
            doc_emb(u_file, i_file, vector_num, domain="book")

        rating_file.to_csv("/data/CrossRec/data/douban_book/ratings_p.csv")
        i_file.to_csv("/data/CrossRec/data/douban_book/book_feature_p.csv")
        u_file.to_csv("/data/CrossRec/data/douban_book/u_feature_p.csv")
    else:
        id_unique = rating_file['music_id'].unique().tolist()
        re_id_music = dict(zip(id_unique, list(range(1,len(id_unique)+1))))
        i_file['UID'] = i_file['UID'].map(re_id_music)
        rating_file['music_id'] = rating_file['music_id'].map(re_id_music)

        u_file = u_file[["UID", "user_info"]]
        i_file = i_file[["UID", "music_info"]]
        rating_file = rating_file[["user_id", "music_id", "rating"]]
        u_file['user_info'] = u_file['user_info'].apply(lambda x: eval(x))
        u_file['user_info'] = u_file['user_info'].apply(lambda x: eval('[%s]' % repr(x).replace('[', '').replace(']', '')))
        i_file["music_info"] = i_file['music_info'].apply(lambda x: eval(x))
        i_file['music_info'] = i_file['music_info'].apply(lambda x: eval('[%s]' % repr(x).replace('[', '').replace(']', '')))
        for vector_num in vector_nums:
            doc_emb(u_file, i_file, vector_num, domain="music")
        rating_file.to_csv("/data/CrossRec/data/douban_music/ratings_p.csv")
        i_file.to_csv("/data/CrossRec/data/douban_music/music_feature_p.csv")
        u_file.to_csv("/data/CrossRec/data/douban_music/u_feature_p.csv")
    print("结束处理{}的数据!".format(domain))


def doc_emb(u_file, i_file, vec_num, domain="movie"):
    print("Start!")
    documents = u_file['user_info'].values.tolist() + \
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
    model.save(root_p + "douban_" + domain + "/Doc2vec_douban_%s_VSize%02d.model" % (domain,vec_num))
    vectors = model.docvecs.vectors_docs
    print("End!")




def read_file(path):
    """读取文件"""
    file = pd.read_table(path)
    return file



def douban():
    """合起来构成douban的特征处理"""
    print("Douban Start!")
    movie_domain()
    book_domain()
    music_domain()
    print("End!")




if __name__ == '__main__':
    # root = "/data/CrossRec/data/douban_feature_process/"
    domains = ['douban_movie','douban_book', 'douban_music']

    # 40324
    # model_D2V_64 = Doc2Vec.load(root + "Doc2vec_" + "douban_book" + "_VSize_" + str(64) + ".model")
    # model_D2V = Doc2Vec.load(root + "Doc2vec_" + "douban_book" + "_VSize_" + str(128) + ".model")
    # print(model_D2V_64.wv.vectors.shape) # 只有前面的能用？？？我没搞懂为什么有4w多条

    # user_size = 2718  # total users in 3 domains
    # movie_size = 9555
    # book_size = 6777
    # music_size = 5567
    # vector_nums = [8, 16, 32, 64, 128]

    # for domain in domains:
    #     for K_Size in vector_nums:
    #         model_D2V = Doc2Vec.load((root_p + domain + "/Doc2vec_" + domain + "_VSize%02d" + ".model") % K_Size)
    #         print(model_D2V.docvecs.vectors_docs.shape) # U +I, wv.vectors是词向量的维度

    # douban()
    #
    # m = pd.read_csv("/data/CrossRec/data/douban_feature/m_feature.csv", index_col=0)
    # um_feature = pd.read_csv("/data/CrossRec/data/douban_feature/um_feature.csv", index_col=0)
    # um_inter = pd.read_csv("/data/CrossRec/data/douban_feature/um_inter.csv", index_col=0)
    # re_id(um_inter, m, um_feature, 'movie',vector_nums)
    #
    # b = pd.read_csv("/data/CrossRec/data/douban_feature/b_feature.csv",index_col=0)
    # ub_feature = pd.read_csv("/data/CrossRec/data/douban_feature/ub_feature.csv",index_col=0)
    # ub_inter = pd.read_csv("/data/CrossRec/data/douban_feature/ub_inter.csv",index_col=0)
    # re_id(ub_inter,b,ub_feature,'book',vector_nums)
    #
    # mu = pd.read_csv("/data/CrossRec/data/douban_feature/mu_feature.csv",index_col=0)
    # umu_feature = pd.read_csv("/data/CrossRec/data/douban_feature/umu_feature.csv",index_col=0)
    # umu_inter = pd.read_csv("/data/CrossRec/data/douban_feature/umu_inter.csv",index_col=0)
    # re_id(umu_inter,mu,umu_feature,'music',vector_nums)

    # item应该排序的。。。
    # data = pd.read_csv("/data/CrossRec/data/douban_music/music_feature_p.csv")
    # print(data["UID"].max())
    # print(len(data["UID"].unique()))
    # print(data['UID'].min())
    # print(data["UID"].head())
    # print(data["UID"])






    # data_r = pd.read_table("/data/CrossRec/data/douban_feature_raw/musicreviews_cleaned.txt")
    # data_map = pd.read_table("/data/CrossRec/data/douban_feature_raw/music_cleaned.txt")
    # print(data_r['music_id'].max())
    # music_list = data_r['music_id'].unique().tolist()
    # print(data_map.shape)
    # data_map = data_map[data_map.UID.isin(music_list)].reset_index(inplace=True)
    # print(data_r.shape)


    # book_domain()


    # data = pd.read_table(root+"moviereviews_cleaned.txt")
    # print(data.shape)
    # data = data.groupby("movie_id").filter(lambda x: (len(x) > 20)) # 用的是20次
    # print(data.shape)
    # print(len(data['movie_id'].unique()))

    # data['useful_num'] = data['useful_num'].astype(int)
    # print(len(data[data['useful_num']>=5]['movie_id'].unique()))

    # data = pd.read_table(root+"bookreviews_cleaned.txt")
    # print(data.shape)
    # data = data.groupby("book_id").filter(lambda x: (len(x) > 5)) # 用的是20次
    # print(data.shape)
    # print(len(data['book_id'].unique()))
    # data = pd.read_table(root + "musicreviews_cleaned.txt")
    # print(data.shape)
    # data = data.groupby("music_id").filter(lambda x: (len(x) > 5))  # 用的是20次
    # print(data.shape)
    # print(len(data['music_id'].unique()))

    # 测试
    # file = pd.read_csv("/data/CrossRec/data/douban_feature/um_inter_d.csv")
    # print(file.shape)

    # 查看重叠
    data_book = pd.read_csv("/data/CrossRec/data/douban_book/ratings_p.csv")
    data_music = pd.read_csv("/data/CrossRec/data/douban_music/ratings_p.csv")
    print(len(set(data_book['user_id'].unique().tolist()) & set(data_music['user_id'].unique().tolist())))



