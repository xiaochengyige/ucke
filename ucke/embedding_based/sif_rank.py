# -*- coding: UTF-8 -*-
import os

import thulac

from ucke.base import CONFIG_PATH, MODELS_PATH
from ucke.register import models
from ucke.utils import SentEmbeddings, WordEmbeddings, sif_rank, sif_rank_plus


@models.register
class SIFRank:
    def __init__(self, lambda_=0.8, elmo_layers_weight=[0.0, 1.0, 0.0]):
        self.elmo_layers_weight = elmo_layers_weight
        self.elmo = WordEmbeddings(os.path.join(
            MODELS_PATH, 'sif_rank\\zhs.model\\'))
        self.sif = SentEmbeddings(self.elmo, weightfile_pretrain=os.path.join(MODELS_PATH, 'sif_rank\\dict.txt'),
                                  weightfile_finetune=os.path.join(MODELS_PATH, 'sif_rank\\dict.txt'), lamda=lambda_)
        self.zh_model = thulac.thulac(model_path=os.path.join(
            MODELS_PATH, 'sif_rank\\thulac.models\\'), user_dict=os.path.join(CONFIG_PATH, 'jieba_user_dict.txt'))

    def tokenize(self, text):
        self.text = text

    def extract_keywords(self, top_k=1):
        sorted_phrases = sif_rank(self.text, self.sif, self.zh_model,
                                  elmo_layers_weight=self.elmo_layers_weight)
        return sorted_phrases[:top_k]


@models.register
class SIFRank_plus:
    def __init__(self, lambda_=0.8, elmo_layers_weight=[0.0, 1.0, 0.0]):
        self.elmo_layers_weight = elmo_layers_weight
        self.elmo = WordEmbeddings(os.path.join(
            MODELS_PATH, 'sif_rank\\zhs.model\\'))
        self.sif = SentEmbeddings(self.elmo, weightfile_pretrain=os.path.join(MODELS_PATH, 'sif_rank\\dict.txt'),
                                  weightfile_finetune=os.path.join(MODELS_PATH, 'sif_rank\\dict.txt'), lamda=lambda_)
        self.zh_model = thulac.thulac(model_path=os.path.join(
            MODELS_PATH, 'sif_rank\\thulac.models\\'), user_dict=os.path.join(CONFIG_PATH, 'jieba_user_dict.txt'))

    def tokenize(self, text):
        self.text = text

    def extract_keywords(self, top_k=1):
        sorted_phrases = sif_rank_plus(self.text, self.sif, self.zh_model,
                                       elmo_layers_weight=self.elmo_layers_weight)
        return sorted_phrases[:top_k]
