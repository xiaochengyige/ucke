# -*- coding: UTF-8 -*-
import networkx

from ucke.base import Base
from ucke.register import models
from ucke.utils import set_graph_edges


@models.register
class TextRank(Base):

    def __init__(self, lambda_=0.85):
        super(TextRank, self).__init__()
        self.lambda_ = lambda_

    def extract_keywords(self, top_k=1):
        words = self.word_list
        graph = networkx.Graph()
        graph.add_nodes_from(set(words))
        set_graph_edges(graph, words, words)

        # 使用默认的PageRank算法评分结点
        ranks = networkx.pagerank(graph, self.lambda_)
        sorted_phrases = sorted(
            ranks.items(), key=lambda x: x[1], reverse=True)

        return sorted_phrases[:top_k]
