# -*- coding: UTF-8 -*-
import copy
import math
from collections import Counter

import numpy as np

from ucke.base import Base
from ucke.register import models


@models.register
class PositionRank(Base):

    def __init__(self, window_size=6, lambda_=0.85):
        super(PositionRank, self).__init__()
        self.window_size = window_size
        self.lambda_ = lambda_

    def weight_total(self, matrix, idx, s_vec):
        """Sum weights of adjacent nodes.

        Choose 'j'th nodes which is adjacent to 'i'th node.
        Sum weight in 'j'th column, then devide wij(weight of index i,j).
        This calculation is applied to all adjacent node, and finally return sum of them.

        """
        return sum([(wij / matrix.sum(axis=0)[j]) * s_vec[j] for j, wij in enumerate(matrix[idx]) if not wij == 0])

    def extract_keywords(self, top_k=1):
        words = self.word_list

        unique_word_list = set([word for word in words])
        n = len(unique_word_list)

        adjancency_matrix = np.zeros((n, n))
        word2idx = {w: i for i, w in enumerate(unique_word_list)}
        p_vec = np.zeros(n)
        # store co-occurence words
        co_occ_dict = {w: [] for w in unique_word_list}

        # 1. initialize  probability vector
        for i, w in enumerate(words):
            # add position score
            p_vec[word2idx[w]] += float(1 / (i + 1))
            for window_idx in range(1, math.ceil(self.window_size / 2) + 1):
                if i - window_idx >= 0:
                    co_list = co_occ_dict[w]
                    co_list.append(words[i - window_idx])
                    co_occ_dict[w] = co_list

                if i + window_idx < len(words):
                    co_list = co_occ_dict[w]
                    co_list.append(words[i + window_idx])
                    co_occ_dict[w] = co_list

        # 2. create adjancency matrix from co-occurence word
        for w, co_list in co_occ_dict.items():
            cnt = Counter(co_list)
            for co_word, freq in cnt.most_common():
                adjancency_matrix[word2idx[w]][word2idx[co_word]] = freq

        adjancency_matrix = adjancency_matrix / adjancency_matrix.sum(axis=0)
        p_vec = p_vec / p_vec.sum()
        # principal eigenvector s
        s_vec = np.ones(n) / n

        # threshold
        lambda_val = 1.0
        loop = 0
        # compute final principal eigenvector
        while lambda_val > 0.001:
            next_s_vec = copy.deepcopy(s_vec)
            for i, (p, s) in enumerate(zip(p_vec, s_vec)):
                next_s = (1 - self.lambda_) * p + self.lambda_ * \
                    (self.weight_total(adjancency_matrix, i, s_vec))
                next_s_vec[i] = next_s
            lambda_val = np.linalg.norm(next_s_vec - s_vec)
            s_vec = next_s_vec
            loop += 1
            if loop > 100:
                break

        # score original words and phrases
        ranks = {word: s_vec[word2idx[word]] for word in unique_word_list}
        sorted_phrases = sorted(
            ranks.items(), key=lambda x: x[1], reverse=True)

        return sorted_phrases[:top_k]
