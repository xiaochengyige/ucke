# -*- coding: UTF-8 -*-
from collections import defaultdict
from itertools import product

import networkx
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.feature_extraction.text import CountVectorizer

from ucke.base import Base
from ucke.register import models


@models.register
class TopicRank(Base):

    def __init__(self, max_d=0.75, lambda_=0.85):
        super(TopicRank, self).__init__()
        self.max_d = max_d
        self.lambda_ = lambda_

    def calc_distance(self, topic_a, topic_b, position_map):
        """
        Calculate distance between 2 topics
        :param topic_a: list if phrases in a topic A
        :param topic_b: list if phrases in a topic B
        :return: int
        """
        result = 0
        for phrase_a in topic_a:
            for phrase_b in topic_b:
                if phrase_a != phrase_b:
                    phrase_a_positions = position_map[phrase_a]
                    phrase_b_positions = position_map[phrase_b]
                    for a, b in product(phrase_a_positions, phrase_b_positions):
                        result += 1 / abs(a - b)
        return result

    def extract_keywords(self, top_k=1):
        words = self.word_list
        position_map = defaultdict(list)

        # get position info
        for idx, word in enumerate(words):
            position_map[word].append(idx)

        # use term freq to convert phrases to vectors for clustering
        count = CountVectorizer()
        bag = count.fit_transform(list(position_map.keys()))

        # apply HAC
        Z = linkage(bag.toarray(), 'average')

        # identify clusters
        clusters = fcluster(Z, self.max_d, criterion='distance')
        cluster_data = defaultdict(list)
        for n, cluster in enumerate(clusters):
            cluster_data[cluster].append(' '.join(
                sorted([str(i) for i in count.inverse_transform(bag.toarray()[n].reshape(1, -1))[0]])))
        topic_clusters = [frozenset(i) for i in cluster_data.values()]

        topic_graph = networkx.Graph()
        topic_graph.add_weighted_edges_from([(v, u, self.calc_distance(
            v, u, position_map)) for v in topic_clusters for u in topic_clusters if u != v])
        ranks = networkx.pagerank(topic_graph, self.lambda_, weight='weight')

        # sort topic by rank
        topics = sorted([(b, list(a)) for a, b in ranks.items()], reverse=True)
        sorted_phrases = [(topic_list[0], score)
                          for score, topic_list in topics]
        return sorted_phrases[:top_k]
