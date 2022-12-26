# -*- coding: UTF-8 -*-
import os
import re

import jieba
import jieba.posseg as psg

from ucke.register import models

ROOT_PATH = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(ROOT_PATH, 'config')
DATA_PATH = os.path.join(ROOT_PATH, 'data')
MODELS_PATH = os.path.join(ROOT_PATH, 'models')
JIEBA_USER_DICT = os.path.join(CONFIG_PATH, 'jieba_user_dict.txt')
STOP_WORDS = os.path.join(CONFIG_PATH, 'stop_words.txt')
POS_DICT = os.path.join(CONFIG_PATH, 'POS_dict.txt')


class Base:
    def __init__(self):
        # 初始化jieba分词
        jieba.load_userdict(JIEBA_USER_DICT)
        jieba.initialize()

        # 加载词性筛选配置
        self.flag_list = []
        with open(POS_DICT, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                self.flag_list.append(line)

        # 加载停留词筛选配置
        try:
            stop_word_list = open(STOP_WORDS, encoding='utf-8')
        except:
            stop_word_list = []
            print("Error in stop_words file")
        self.stop_list = []
        for line in stop_word_list:
            line = re.sub(u'\n|\\r', '', line)
            self.stop_list.append(line)

        self.word_list = []

    def tokenize(self, text):
        seg_list = psg.cut(text)
        for seg_word in seg_list:
            word = re.sub(u'[^\u4e00-\u9fa5]', '', seg_word.word)
            find = 0
            for stop_word in self.stop_list:
                if stop_word == word or len(word) < 2:  # this word is stopword
                    find = 1
                    break
            if find == 0 and seg_word.flag in self.flag_list:
                self.word_list.append(word)

    def extract_keywords(self, top_k=1):
        return []
