# -*- coding: UTF-8 -*-
import os

from ucke.base import MODELS_PATH, Base
from ucke.register import models


@models.register
class TFIDF(Base):

    def __init__(self):
        super(TFIDF, self).__init__()
        content = open(os.path.join(MODELS_PATH, "tfidf\\idf.txt"),
                       'rb').read().decode('utf-8')
        self.idf_freq = {}
        for line in content.splitlines():
            word, freq = line.strip().split(' ')
            self.idf_freq[word] = float(freq)
        self.median_idf = sorted(
            self.idf_freq.values())[len(self.idf_freq) // 2]

    def extract_keywords(self, top_k=1):
        words = self.word_list
        freq = {}
        for word in words:
            freq[word] = freq.get(word, 0.0) + 1.0
        total = sum(freq.values())
        for key in freq:
            freq[key] *= self.idf_freq.get(key, self.median_idf) / total

        sorted_phrases = sorted(freq, key=freq.__getitem__, reverse=True)
        return sorted_phrases[:top_k]
