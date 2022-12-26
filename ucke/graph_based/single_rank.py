# -*- coding: UTF-8 -*-
import networkx

from ucke.base import Base
from ucke.register import models


@models.register
class SingleRank(Base):

    def __init__(self, window=10, lambda_=0.85):
        super(SingleRank, self).__init__()
        self.window = window
        self.lambda_ = lambda_

    def extract_keywords(self, top_k=1):
        words = self.word_list
        graph = networkx.Graph()
        # add nodes to the graph
        graph.add_nodes_from(set(words))
        # add edges to the graph
        for i, node1 in enumerate(words):
            for j in range(i + 1, min(i + self.window, len(words))):
                node2 = words[j]
                if node1 != node2:
                    if not graph.has_edge(node1, node2):
                        graph.add_edge(node1, node2, weight=0.0)
                    graph[node1][node2]['weight'] += 1.0

        # compute the word scores using random walk
        ranks = networkx.pagerank(graph, self.lambda_)

        sorted_phrases = sorted(
            ranks.items(), key=lambda x: x[1], reverse=True)

        return sorted_phrases[:top_k]
