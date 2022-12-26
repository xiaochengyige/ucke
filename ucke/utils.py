# -*- coding: UTF-8 -*-
from itertools import combinations as combinations
from queue import Queue

import nltk
import numpy as np
import torch
from elmoformanylangs import Embedder
from nltk.corpus import stopwords

english_punctuations = [',', '.', ':', ';', '?',
                        '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
chinese_punctuations = '！？｡。＂＃＄％＆＇（）＊＋,－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.'

stop_words = set(stopwords.words("english"))
wnl = nltk.WordNetLemmatizer()
considered_tags = {'n', 'np', 'ns', 'ni', 'nz', 'a', 'd', 'i', 'j', 'x', 'g'}

GRAMMAR_zh = """  NP:
        {<n.*|a|uw|i|j|x>*<n.*|uw|x>|<x|j><-><m|q>} # Adjective(s)(optional) + Noun(s)"""

WINDOW_SIZE = 2


def get_first_window(split_text):
    return split_text[:WINDOW_SIZE]


# tokens is a list of words
def set_graph_edge(graph, tokens, word_a, word_b):
    if word_a in tokens and word_b in tokens:
        edge = (word_a, word_b)
        if graph.has_node(word_a) and graph.has_node(word_b) and not graph.has_edge(*edge):
            graph.add_edge(*edge)


def process_first_window(graph, tokens, split_text):
    first_window = get_first_window(split_text)
    for word_a, word_b in combinations(first_window, 2):
        set_graph_edge(graph, tokens, word_a, word_b)


def init_queue(split_text):
    queue = Queue()
    first_window = get_first_window(split_text)
    for word in first_window[1:]:
        queue.put(word)
    return queue


def queue_iterator(queue):
    iterations = queue.qsize()
    for i in range(iterations):
        var = queue.get()
        yield var
        queue.put(var)


def process_word(graph, tokens, queue, word):
    for word_to_compare in queue_iterator(queue):
        set_graph_edge(graph, tokens, word, word_to_compare)


def update_queue(queue, word):
    queue.get()
    queue.put(word)
    assert queue.qsize() == (WINDOW_SIZE - 1)


def process_text(graph, tokens, split_text):
    queue = init_queue(split_text)
    for i in range(WINDOW_SIZE, len(split_text)):
        word = split_text[i]
        process_word(graph, tokens, queue, word)
        update_queue(queue, word)


def set_graph_edges(graph, tokens, split_text):
    process_first_window(graph, tokens, split_text)
    process_text(graph, tokens, split_text)


def extract_candidates(tokens_tagged):
    """
    Based on part of speech return a list of candidate phrases
    :param text_obj: Input text Representation see @InputTextObj
    :param no_subset: if true won't put a candidate which is the subset of an other candidate
    :return keyphrase_candidate: list of list of candidate phrases: [tuple(string,tuple(start_index,end_index))]
    """
    np_parser = nltk.RegexpParser(GRAMMAR_zh)  # Noun phrase parser
    keyphrase_candidate = []
    np_pos_tag_tokens = np_parser.parse(tokens_tagged)
    count = 0
    for token in np_pos_tag_tokens:
        if isinstance(token, nltk.tree.Tree) and token._label == "NP":
            np = ''.join(word for word, tag in token.leaves())
            length = len(token.leaves())
            start_end = (count, count + length)
            count += length
            keyphrase_candidate.append((np, start_end))

        else:
            count += 1

    return keyphrase_candidate


class InputTextObj:
    """Represent the input text in which we want to extract keyphrases"""

    def __init__(self, zh_model, text=""):
        """
        :param is_sectioned: If we want to section the text.
        :param zh_model: the pipeline of Chinese tokenization and POS-tagger
        :param considered_tags: The POSs we want to keep
        """
        self.considered_tags = {'n', 'np', 'ns',
                                'ni', 'nz', 'a', 'd', 'i', 'j', 'x', 'g'}

        self.tokens = []
        self.tokens_tagged = []
        word_pos = zh_model.cut(text)
        self.tokens = [word_pos[0] for word_pos in word_pos]
        self.tokens_tagged = [(word_pos[0], word_pos[1])
                              for word_pos in word_pos]
        assert len(self.tokens) == len(self.tokens_tagged)
        for i, token in enumerate(self.tokens):
            if token.lower() in stop_words:
                self.tokens_tagged[i] = (token, "u")
            if token == '-':
                self.tokens_tagged[i] = (token, "-")
        self.keyphrase_candidate = extract_candidates(self.tokens_tagged)


def cos_sim_gpu(x, y):
    assert x.shape[0] == y.shape[0]
    zero_tensor = torch.zeros((1, x.shape[0])).cuda()
    # zero_list = [0] * len(x)
    if x == zero_tensor or y == zero_tensor:
        return float(1) if x == y else float(0)
    xx, yy, xy = 0.0, 0.0, 0.0
    for i in range(x.shape[0]):
        xx += x[i] * x[i]
        yy += y[i] * y[i]
        xy += x[i] * y[i]
    return 1.0 - xy / np.sqrt(xx * yy)


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denom == 0.0:
        return 0.0
    else:
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim


def cos_sim_transformer(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    a = vector_a.detach().numpy()
    b = vector_b.detach().numpy()
    a = np.mat(a)
    b = np.mat(b)
    num = float(a * b.T)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    else:
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim


def get_dist_cosine(emb1, emb2, sent_emb_method="elmo", elmo_layers_weight=[0.0, 1.0, 0.0]):
    sum = 0.0
    assert emb1.shape == emb2.shape
    if sent_emb_method == "elmo":
        for i in range(0, 3):
            a = emb1[i]
            b = emb2[i]
            sum += cos_sim(a, b) * elmo_layers_weight[i]
        return sum
    elif sent_emb_method == "elmo_transformer":
        sum = cos_sim_transformer(emb1, emb2)
        return sum
    elif sent_emb_method == "doc2vec":
        sum = cos_sim(emb1, emb2)
        return sum
    return sum


def get_all_dist(candidate_embeddings_list, text_obj, dist_list):
    '''
    :param candidate_embeddings_list:
    :param text_obj:
    :param dist_list:
    :return: dist_all
    '''

    dist_all = {}
    for i, emb in enumerate(candidate_embeddings_list):
        phrase = text_obj.keyphrase_candidate[i][0]
        phrase = phrase.lower()
        phrase = wnl.lemmatize(phrase)
        if phrase in dist_all:
            # store the No. and distance
            dist_all[phrase].append(dist_list[i])
        else:
            dist_all[phrase] = []
            dist_all[phrase].append(dist_list[i])
    return dist_all


def get_final_dist(dist_all, method="average"):
    '''
    :param dist_all:
    :param method: "average"
    :return:
    '''
    final_dist = {}
    if method == "average":
        for phrase, dist_list in dist_all.items():
            sum_dist = 0.0
            for dist in dist_list:
                sum_dist += dist
            if phrase in stop_words:
                sum_dist = 0.0
            final_dist[phrase] = sum_dist / float(len(dist_list))
        return final_dist


def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def get_position_score(keyphrase_candidate_list, position_bias):
    position_score = {}
    for i, kc in enumerate(keyphrase_candidate_list):
        np = kc[0]
        np = np.lower()
        np = wnl.lemmatize(np)
        if np in position_score:
            position_score[np] += 0.0
        else:
            position_score[np] = 1 / (float(i) + 1 + position_bias)
    score_list = []
    for np, score in position_score.items():
        score_list.append(score)
    score_list = softmax(score_list)
    i = 0
    for np, score in position_score.items():
        position_score[np] = score_list[i]
        i += 1
    return position_score


def sif_rank(text, SIF, cn_model, sent_emb_method="elmo", elmo_layers_weight=[0.0, 1.0, 0.0], if_DS=True,
             if_EA=True):
    """

    @param text:
    @param SIF:
    @param cn_model:
    @param sent_emb_method:
    @param elmo_layers_weight:
    @param if_DS:
    @param if_EA:
    @return:
    """
    text_obj = InputTextObj(cn_model, text)
    sent_embeddings, candidate_embeddings_list = SIF.get_tokenized_sent_embeddings(
        text_obj, if_DS=if_DS, if_EA=if_EA)
    dist_list = []
    for i, emb in enumerate(candidate_embeddings_list):
        dist = get_dist_cosine(
            sent_embeddings, emb, sent_emb_method, elmo_layers_weight=elmo_layers_weight)
        dist_list.append(dist)
    dist_all = get_all_dist(candidate_embeddings_list, text_obj, dist_list)
    dist_final = get_final_dist(dist_all, method='average')
    dist_sorted = sorted(dist_final.items(), key=lambda x: x[1], reverse=True)
    return dist_sorted


def sif_rank_plus(text, SIF, cn_model, sent_emb_method="elmo", elmo_layers_weight=[0.0, 1.0, 0.0], if_DS=True,
                  if_EA=True, position_bias=3.4):
    """

    @param text:
    @param SIF:
    @param cn_model:
    @param sent_emb_method:
    @param elmo_layers_weight:
    @param if_DS:
    @param if_EA:
    @param position_bias:
    @return:
    """
    text_obj = InputTextObj(cn_model, text)
    sent_embeddings, candidate_embeddings_list = SIF.get_tokenized_sent_embeddings(
        text_obj, if_DS=if_DS, if_EA=if_EA)
    position_score = get_position_score(
        text_obj.keyphrase_candidate, position_bias)
    average_score = sum(position_score.values()) / (float)(len(position_score))

    dist_list = []
    for i, emb in enumerate(candidate_embeddings_list):
        dist = get_dist_cosine(
            sent_embeddings, emb, sent_emb_method, elmo_layers_weight=elmo_layers_weight)
        dist_list.append(dist)
    dist_all = get_all_dist(candidate_embeddings_list, text_obj, dist_list)
    dist_final = get_final_dist(dist_all, method='average')

    for np, dist in dist_final.items():
        if np in position_score:
            dist_final[np] = dist * position_score[np] / average_score
    dist_sorted = sorted(dist_final.items(), key=lambda x: x[1], reverse=True)
    return dist_sorted


class WordEmbeddings:
    """
        ELMo
        https://allennlp.org/elmo

    """

    def __init__(self, model_path=r'./zhs.model/', cuda_device=0):
        self.cuda_device = cuda_device
        self.elmo = Embedder(model_path)

    def get_tokenized_words_embeddings(self, sents_tokened):
        """
        @param sents_tokened: list of tokenized words string (sentences/phrases)
        :return: ndarray with shape (len(sents), dimension of embeddings)
        """
        max_len = max([len(sent) for sent in sents_tokened])
        elmo_embedding = self.elmo.sents2elmo(sents_tokened, output_layer=-2)
        elmo_embedding = [np.pad(emb, pad_width=((0, 0), (0, max_len - emb.shape[1]), (0, 0)), mode='constant') for emb
                          in elmo_embedding]
        elmo_embedding = torch.from_numpy(np.array(elmo_embedding))
        return elmo_embedding


class SentEmbeddings:
    def __init__(self, word_embeddor, weightfile_pretrain='./model/SIF_rank/dict.txt', weightfile_finetune='./dict.txt',
                 weightpara_pretrain=2.7e-4, weightpara_finetune=2.7e-4, lamda=1.0, database="",
                 embeddings_type="elmo"):
        self.word2weight_pretrain = get_word_weight(
            weightfile_pretrain, weightpara_pretrain)
        self.word2weight_finetune = get_word_weight(
            weightfile_finetune, weightpara_finetune)
        self.word_embeddor = word_embeddor
        self.lamda = lamda
        self.database = database
        self.embeddings_type = embeddings_type

    def get_tokenized_sent_embeddings(self, text_obj, if_DS=False, if_EA=False):
        """
        Based on part of speech return a list of candidate phrases
        :param text_obj: Input text Representation see @InputTextObj
        :param if_DS: if take document segmentation(DS)
        :param if_EA: if take  embeddings alignment(EA)
        """
        # choose the type of word embeddings:elmo
        if self.embeddings_type == "elmo" and if_DS == False:
            elmo_embeddings = self.word_embeddor.get_tokenized_words_embeddings([
                                                                                text_obj.tokens])
        elif self.embeddings_type == "elmo" and if_DS == True and if_EA == False:
            tokens_segmented = get_sent_segmented(text_obj.tokens)
            elmo_embeddings = self.word_embeddor.get_tokenized_words_embeddings(
                tokens_segmented)
            elmo_embeddings = splice_embeddings(
                elmo_embeddings, tokens_segmented)
        elif self.embeddings_type == "elmo" and if_DS == True and if_EA == True:
            tokens_segmented = get_sent_segmented(text_obj.tokens)
            elmo_embeddings = self.word_embeddor.get_tokenized_words_embeddings(
                tokens_segmented)
            elmo_embeddings = context_embeddings_alignment(
                elmo_embeddings, tokens_segmented)
            elmo_embeddings = splice_embeddings(
                elmo_embeddings, tokens_segmented)
        else:
            elmo_embeddings, elmo_mask = self.word_embeddor.get_tokenized_words_embeddings(
                text_obj.tokens)

        candidate_embeddings_list = []

        weight_list = get_weight_list(self.word2weight_pretrain, self.word2weight_finetune, text_obj.tokens,
                                      lamda=self.lamda, database=self.database)

        sent_embeddings = get_weighted_average(text_obj.tokens, text_obj.tokens_tagged, weight_list, elmo_embeddings[0],
                                               embeddings_type=self.embeddings_type)

        for kc in text_obj.keyphrase_candidate:
            start = kc[1][0]
            end = kc[1][1]
            kc_emb = get_candidate_weighted_average(text_obj.tokens, weight_list, elmo_embeddings[0], start, end,
                                                    embeddings_type=self.embeddings_type)
            candidate_embeddings_list.append(kc_emb)

        return sent_embeddings, candidate_embeddings_list


def context_embeddings_alignment(elmo_embeddings, tokens_segmented):
    """
    Embeddings Alignment
    :param elmo_embeddings: The embeddings from elmo
    :param tokens_segmented: The list of tokens list
     <class 'list'>: [['今', '天', '天气', '真', '好', '啊'],['潮水', '退', '了', '就', '知道', '谁', '没', '穿', '裤子']]
    :return:
    """
    token_emb_map = {}
    n = 0
    for i in range(0, len(tokens_segmented)):

        for j, token in enumerate(tokens_segmented[i]):

            emb = elmo_embeddings[i, 1, j, :]
            if token not in token_emb_map:
                token_emb_map[token] = [emb]
            else:
                token_emb_map[token].append(emb)
            n += 1

    anchor_emb_map = {}
    for token, emb_list in token_emb_map.items():
        average_emb = emb_list[0]
        for j in range(1, len(emb_list)):
            average_emb += emb_list[j]
        average_emb /= float(len(emb_list))
        anchor_emb_map[token] = average_emb

    for i in range(0, elmo_embeddings.shape[0]):
        for j, token in enumerate(tokens_segmented[i]):
            emb = anchor_emb_map[token]
            elmo_embeddings[i, 2, j, :] = emb

    return elmo_embeddings


def mat_division(vector_a, vector_b):
    a = vector_a.detach().numpy()
    b = vector_b.detach().numpy()
    A = np.mat(a)
    B = np.mat(b)
    return torch.from_numpy(np.dot(A.I, B))


def get_sent_segmented(tokens):
    min_seq_len = 16
    sents_sectioned = []
    if len(tokens) <= min_seq_len:
        sents_sectioned.append(tokens)
    else:
        position = 0
        for i, token in enumerate(tokens):
            if token == '.' or token == '。':
                if i - position >= min_seq_len:
                    sents_sectioned.append(tokens[position:i + 1])
                    position = i + 1
        if len(tokens[position:]) > 0:
            sents_sectioned.append(tokens[position:])

    return sents_sectioned


def splice_embeddings(elmo_embeddings, tokens_segmented):
    new_elmo_embeddings = elmo_embeddings[0:1, :, 0:len(
        tokens_segmented[0]), :]
    for i in range(1, len(tokens_segmented)):
        emb = elmo_embeddings[i:i + 1, :, 0:len(tokens_segmented[i]), :]
        new_elmo_embeddings = torch.cat((new_elmo_embeddings, emb), 2)
    return new_elmo_embeddings


def get_effective_words_num(tokened_sents):
    i = 0
    for token in tokened_sents:
        if token not in english_punctuations:
            i += 1
    return i


def get_weighted_average(tokenized_sents, sents_tokened_tagged, weight_list, embeddings_list, embeddings_type="elmo"):
    assert len(tokenized_sents) == len(weight_list)
    num_words = len(tokenized_sents)
    e_test_list = []
    if embeddings_type == "elmo" or embeddings_type == "elmo_sectioned":
        sum = torch.zeros((3, 1024))
        for i in range(0, 3):
            for j in range(0, num_words):
                if sents_tokened_tagged[j][1] in considered_tags:
                    e_test = embeddings_list[i][j]
                    e_test_list.append(e_test)
                    sum[i] += e_test * weight_list[j]

            sum[i] = sum[i] / float(num_words)
        return sum
    elif embeddings_type == "elmo_transformer":
        sum = torch.zeros((1, 1024))
        for i in range(0, 1):
            for j in range(0, num_words):
                if sents_tokened_tagged[j][1] in considered_tags:
                    e_test = embeddings_list[i][j]
                    e_test_list.append(e_test)
                    sum[i] += e_test * weight_list[j]
            sum[i] = sum[i] / float(num_words)
        return sum
    return 0


def get_candidate_weighted_average(tokenized_sents, weight_list, embeddings_list, start, end, embeddings_type="elmo"):
    assert len(tokenized_sents) == len(weight_list)
    num_words = end - start
    e_test_list = []
    if embeddings_type == "elmo" or embeddings_type == "elmo_sectioned":
        sum = torch.zeros((3, 1024))
        for i in range(0, 3):
            for j in range(start, end):
                e_test = embeddings_list[i][j]
                e_test_list.append(e_test)
                sum[i] += e_test * weight_list[j]
            sum[i] = sum[i] / float(num_words)

        return sum
    elif embeddings_type == "elmo_transformer":
        sum = torch.zeros((1, 1024))
        for i in range(0, 1):
            for j in range(start, end):
                e_test = embeddings_list[i][j]
                e_test_list.append(e_test)
                sum[i] += e_test * weight_list[j]
            sum[i] = sum[i] / float(num_words)
        return sum
    return 0


def get_oov_weight(tokenized_sents, word2weight, word, method="max_weight"):
    word = wnl.lemmatize(word)

    if word in word2weight:
        return word2weight[word]
    if word in stop_words:
        return 0.0
    if word in english_punctuations or word in chinese_punctuations:  # The oov_word is a punctuation
        return 0.0
    if method == "max_weight":  # Return the max weight of word in the tokenized_sents
        max = 0.0
        for w in tokenized_sents:
            if w in word2weight and word2weight[w] > max:
                max = word2weight[w]
        return max
    return 0.0


def get_weight_list(word2weight_pretrain, word2weight_finetune, tokenized_sents, lamda, database=""):
    weight_list = []
    for word in tokenized_sents:
        word = word.lower()
        if database == "":
            weight_pretrain = get_oov_weight(
                tokenized_sents, word2weight_pretrain, word, method="max_weight")
            weight = weight_pretrain
        else:
            weight_pretrain = get_oov_weight(
                tokenized_sents, word2weight_pretrain, word, method="max_weight")
            weight_finetune = get_oov_weight(
                tokenized_sents, word2weight_finetune, word, method="max_weight")
            weight = lamda * weight_pretrain + (1.0 - lamda) * weight_finetune
        weight_list.append(weight)

    return weight_list


def get_normalized_weight(weight_list):
    sum_weight = 0.0
    for weight in weight_list:
        sum_weight += weight
    if sum_weight == 0.0:
        return weight_list
    for i in range(0, len(weight_list)):
        weight_list[i] /= sum_weight
    return weight_list


def get_word_weight(weightfile="", weightpara=2.7e-4):
    """
    Get the weight of words by word_fre/sum_fre_words
    :param weightfile
    :param weightpara
    :return: word2weight[word]=weight : a dict of word weight
    """
    if weightpara <= 0:  # when the parameter makes no sense, use unweighted
        weightpara = 1.0
    word2weight = {}
    word2fre = {}
    with open(weightfile, encoding='UTF-8') as f:
        lines = f.readlines()
    sum_fre_words = 0
    for line in lines:
        word_fre = line.split()
        if len(word_fre) >= 2:
            word2fre[word_fre[0]] = float(word_fre[1])
            sum_fre_words += float(word_fre[1])
        else:
            print(line)
    for key, value in word2fre.items():
        word2weight[key] = weightpara / (weightpara + value / sum_fre_words)
    return word2weight
