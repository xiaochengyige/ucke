# -*- coding: UTF-8 -*-
import math

import gensim
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

from ucke.base import Base
from ucke.register import models


@models.register
class Word2Vector(Base):

    def __init__(self):
        super(Word2Vector, self).__init__()

    def extract_keywords(self, top_k=1):
        words = self.word_list
        # ## 模型训练
        # 调用Word2Vec训练
        # 参数：size: 词向量维度；window: 上下文的宽度，min_count为考虑计算的单词的最低词频阈值
        model = Word2Vec(words, size=20, window=2, min_count=1,
                         iter=7, negative=10, sg=0)
        # 获取model里面的所有关键词
        keys = model.wv.vocab.keys()
        wordvector = []
        for key in keys:
            wordvector.append(model.wv.get_vector(key))

        # ## 对词向量采用K-means聚类抽取TopK关键词
        classCount = 1  # 分类数
        clf = KMeans(n_clusters=classCount)
        s = clf.fit(wordvector)
        labels = clf.labels_
        vec_center = clf.cluster_centers_  # 聚类中心
        distances = []
        vec_words = np.array(wordvector)  # 候选关键词向量，dataFrame转array
        vec_center = vec_center[0]  # 第一个类别聚类中心,本例只有一个类别
        length = len(vec_center)  # 向量维度
        for index in range(len(vec_words)):  # 候选关键词个数
            cur_wordvec = vec_words[index]  # 当前词语的词向量
            dis = 0  # 向量距离
            for index2 in range(length):
                dis += (vec_center[index2]-cur_wordvec[index2]) * \
                    (vec_center[index2]-cur_wordvec[index2])
            dis = math.sqrt(dis)
            distances.append(dis)

        sorted_phrases = dict(zip(keys, distances))  # 拼接词语与其对应中心点的距离
        # 按照距离大小进行升序排序
        sorted_phrases = sorted(sorted_phrases.items(),
                                key=lambda x: x[1], reverse=True)

        return sorted_phrases[:top_k]
