# Unsupervised Chinese Keywords Extraction Algorithm

基于**无监督学习**的中文关键词抽取

- 基于统计：TF-IDF

- 基于图：SingleRank, TextRank, TopicRank, PositionRank

- 基于嵌入：Word2Vector, EmbeddingRank, SIFRank, SIFRank+, PatternRank

## Introduction

### Statistics-based

| Algorithm |                               Intro                               | Year |                                                       ref                                                        |
| :-------: | :---------------------------------------------------------------: | :--: | :--------------------------------------------------------------------------------------------------------------: |
|  TF-IDF   | 一种用于信息检索与数据挖掘的常用加权技术,常用于挖掘文章中的关键词 | 1972 |                               [link](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)                               |
|   YAKE    |        首次将主题（Topic）信息整合到 PageRank 计算的公式中        | 2018 | [paper](https://repositorio.inesctec.pt/server/api/core/bitstreams/ef121a01-a0a6-4be8-945d-3324a58fc944/content) |

### Graph-based

|     Algorithm      |                                 Intro                                  | Year |                                  ref                                  |
| :----------------: | :--------------------------------------------------------------------: | :--: | :-------------------------------------------------------------------: |
|      TextRank      |               基于统计,将 PageRank 应用于文本关键词抽取                | 2004 |            [paper](https://aclanthology.org/W04-3252.pdf)             |
|     SingleRank     |            基于统计,TextRank 的一个扩展,它将权重合并到边上             | 2008 |     [paper](https://www.aaai.org/Papers/AAAI/2008/AAAI08-136.pdf)     |
|       SGRank       |                   基于统计,利用了统计和单词共现信息                    | 2015 |            [paper](https://aclanthology.org/S15-1013.pdf)             |
| PositionRank（PR） |         基于统计,利用了单词-单词共现及其在文本中的相应位置信息         | 2017 |            [paper](https://aclanthology.org/P17-1102.pdf)             |
|     ExpandRank     | 基于类似文件/引文网络,SingleRank 扩展,考虑了从相邻文档到目标文档的信息 | 2008 |     [paper](https://www.aaai.org/Papers/AAAI/2008/AAAI08-136.pdf)     |
|    CiteTextRank    |    基于类似文件/引文网络,通过引文网络找到与目标文档更相关的知识背景    | 2014 |    [paper](https://ojs.aaai.org/index.php/AAAI/article/view/8946)     |
|  TopicRank（TR）   |          基于主题,使用层次聚集聚类将候选短语分组为单独的主题           | 2013 |    [paper](https://hal.archives-ouvertes.fr/hal-00917969/document)    |
|        TPR         |      基于主题,首次将主题（Topic）信息整合到 PageRank 计算的公式中      | 2010 |            [paper](https://aclanthology.org/D10-1036.pdf)             |
|     Single TPR     |                 基于主题,单词迭代计算的 Topic PageRank                 | 2015 | [paper](https://biblio.ugent.be/publication/5974208/file/5974209.pdf) |
|   Salience Rank    |                  基于主题,引入显著性的 Topic PageRank                  | 2017 |              [paper](https://aclanthology.org/P17-2084/)              |
|      WikiRank      |            基于语义,构建一个语义图,试图将语义与文本联系起来            | 2018 |             [paper](https://arxiv.org/pdf/1803.09000.pdf)             |

### Embedding-based

|            Algorithm             |                                   Intro                                   | Year |                                    ref                                    |
| :------------------------------: | :-----------------------------------------------------------------------: | :--: | :-----------------------------------------------------------------------: |
|          EmbeddingRank           | 使用句子嵌入（Doc2Vec 或 Sent2vec）在同一高维向量空间中表示候选短语和文档 | 2018 |               [paper](https://arxiv.org/pdf/1801.04470.pdf)               |
| Reference Vector Algorithm (RVA) |      使用局部单词嵌入/语义（Glove）,即从考虑中的单个文档中训练的嵌入      | 2018 |               [paper](https://arxiv.org/pdf/1710.07503.pdf)               |
|         SIFRank/SIFRank+         |                基于预训练语言模型的无监督关键词提取新基线                 | 2020 | [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8954611) |

## Dependencies

```txt
scikit-learn
nltk==3.6.7
gensim==3.8.0
scipy==1.9.3
jieba==0.42.1
networkx==2.5
numpy==1.19.5
pandas==1.1.5
thulac
elmoformanylangs
spacy==3.4.2
xlrd==1.2.0
overrides==3.1.0
ipykernel
```

## File

- `ucke/`：算法主目录
- `ucke/config/`：配置目录
- `ucke/data/`：数据目录
- `ucke/model/`：模型目录

## Data

本项目选取 CSL-40k 数据集中的部分数据作为数据集,并在`ucke/data.py`中提供了该数据集的接口。

## Config

`ucke/config`目录下可以配置：

- `jieba`分词库的自定义词典`jieba_user_dict.txt`,具体参考：[Jieba](https://github.com/fxsjy/jieba#%E8%BD%BD%E5%85%A5%E8%AF%8D%E5%85%B8)
- 添加停用词（`stopwords`）`stop_words.txt`
- 添加词性配置`POS_dict.txt`,即设置提取最终关键词的词性筛选,具体词性表参考：[词性表](https://blog.csdn.net/Yellow_python/article/details/83991967)

如果需要使用`SIF_rank`算法,需要加载`elmo`模型和`thulac`模型：

- `elmo`模型的下载地址：[这里](https://github.com/HIT-SCIR/ELMoForManyLangs),具体位置: `ucke/models/sif_rank/zhs.model`
- `thulac`模型下载地址：[这里](http://thulac.thunlp.org/),具体位置：`ucke/models/sif_rank/thulac.models`
- 百度网盘地址：[这里](https://pan.baidu.com/s/1u2ZS7yFBQULjruvJ2VmbUg),提取码：ucke

## Config

## Usage

### Install

```shell
git clone https://github.com/xiaochengyige/ucke.git

cd ucke

conda create -n ucke python=3.8

conda activate ucke

pip install -r requirements.txt
```

### Run

参考 jupyter 文件`test.ipynb`,该文件展示了本项目所有实现的函数功能

## FAQ

本项目实现了`SIF_rank`算法,该模块用到了`nltk`包,如果你无法根据该包获取`stopwords`或者关于该包的一些其他问题,你可以：

- 前往 [nltk_data](https://github.com/nltk/nltk_data),下载该仓库
- 通过比较可以发现压缩包中的文件结构和`%appdata%/Roaming/nltk_data`下的目录结构是一致的,所以把`packages`里面的东西复制到`%appdata%/Roaming/nltk_data`里面就可以了
- 百度网盘地址：[这里](https://pan.baidu.com/s/1vnurSUIvDeC40gnPNKR3jQ),提取码：ucke
