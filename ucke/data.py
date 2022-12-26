# -*- coding: UTF-8 -*-
import os

import pandas as pd

from ucke.base import DATA_PATH
from ucke.register import init_method


class KEDataset:
    def __init__(self):
        self.data_path = os.path.join(DATA_PATH, 'data.xlsx')
        self.data = pd.read_excel(self.data_path)
        self.methods = {
            ''
        }

    def get_data(self, index=None):
        if index == None:
            return self.data
        else:
            assert isinstance(index, int), '索引必须为整数！'
            assert index >= 0 and index < len(self.data), '索引必须有效！'
            return self.data.iloc[index]

    def __len__(self):
        return len(self.data)

    def evaluate(self, index: int = 0, method: str = 'SingleRank', **kwargs):
        assert index >= 0 and index < len(self.data), '索引必须有效！'
        data = self.data.iloc[index]
        text = data.content
        keywords = data.keywords.split('_')
        algorithm = init_method(method, **kwargs)
        algorithm.tokenize(text)
        extracted_keywords = algorithm.extract_keywords(len(keywords))
        if isinstance(extracted_keywords[0], tuple):
            extracted_keywords = [words[0] for words in extracted_keywords]
        return len(list(set(extracted_keywords) & set(keywords))) / len(keywords)
