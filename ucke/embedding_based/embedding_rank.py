# -*- coding: UTF-8 -*-
import math
import os

import jieba.posseg as jp
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity

from ucke.base import MODELS_PATH, Base
from ucke.register import models


@models.register
class EmbeddingRank(Base):

    def __init__(self, lambda_=0.5):
        super(EmbeddingRank, self).__init__()
        path = os.path.join(
            MODELS_PATH, 'embeddingrank\\embed_rank_doc2vec.bin')
        assert os.path.exists(path), '请训练或下载 EmbeddingRank 预训练模型!'
        self.model = Doc2Vec.load(path)
        self.lambda_ = lambda_
        self.phrases = []
        self.embeddings = []

    def tokenize(self, text):
        super().tokenize(text)
        for w, pos in jp.lcut(text):
            if any(p in pos for p in self.flag_list):
                self.phrases.append(w)
                vector = self.model.infer_vector([w])
                self.embeddings.append(vector)

    def extract_keywords(self, top_k=1):
        phrase_ids, phrases, similarities = self._mmr(self.lambda_)
        if len(phrases) == 0 or len(phrase_ids) == 0 or len(similarities) == 0:
            return []

        sorted_phrases = set()
        for idx in phrase_ids:
            sorted_phrases.add((phrases[idx], similarities[idx][0]))
        sorted_phrases = dict(sorted_phrases)
        sorted_phrases = sorted(sorted_phrases.items(),
                                key=lambda x: math.fabs(x[1]), reverse=True)
        return sorted_phrases[:top_k]

    def _mmr(self, lambda_):
        tokens = self.word_list
        if len(tokens) == 0:
            return [], [], []
        document_embedding = self.model.infer_vector(tokens)
        phrases, phrase_embeddings = self.phrases, self.embeddings
        if len(phrases) == 0:
            return [], [], []
        # shape (num_phrases, embedding_size)
        phrase_embeddings = np.array(phrase_embeddings)
        N = len(phrases)
        # similarity between each phrase and document
        phrase_document_similarities = cosine_similarity(
            phrase_embeddings, document_embedding.reshape(1, -1))
        # similarity between phrases
        phrase_phrase_similarities = cosine_similarity(phrase_embeddings)

        # MMR
        # 1st iteration
        unselected = list(range(len(phrases)))
        # most similiar phrase of document
        select_idx = np.argmax(phrase_document_similarities)

        selected = [select_idx]
        unselected.remove(select_idx)

        # other iterations
        for _ in range(N - 1):
            mmr_distance_to_doc = phrase_document_similarities[unselected, :]
            mmr_distance_between_phrases = np.max(
                phrase_phrase_similarities[unselected][:, selected], axis=1)

            mmr = lambda_ * mmr_distance_to_doc - \
                (1 - lambda_) * mmr_distance_between_phrases.reshape(-1, 1)
            mmr_idx = unselected[np.argmax(mmr)]

            selected.append(mmr_idx)
            unselected.remove(mmr_idx)

        return selected, phrases, phrase_document_similarities
