# -*- coding: UTF-8 -*-
import logging
import os
import warnings
from itertools import combinations as combinations
from typing import List, Union

import nltk
import numpy as np
import psutil
import scipy.sparse as sp
import spacy
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.utils.deprecation import deprecated
from sklearn.utils.validation import FLOAT_DTYPES

from ucke.register import models


@models.register
class PatternRank:
    def __init__(self):
        self.vectorizer = None

    def tokenize(self, text):
        self.vectorizer = KeyphraseCountVectorizer(
            spacy_pipeline='zh_core_web_trf', pos_pattern='<ADJ.*>*<N.*>+', stop_words='chinese')
        self.vectorizer.fit([text])

    def extract_keywords(self, top_k=1):
        return self.vectorizer.get_feature_names()[:top_k]


class _KeyphraseVectorizerMixin():
    """
    _KeyphraseVectorizerMixin

    Provides common code for text vectorizers.
    """

    def _document_frequency(self, document_keyphrase_count_matrix: List[List[int]]) -> np.array:
        """
        Count the number of non-zero values for each feature in sparse a matrix.

        Parameters
        ----------
        document_keyphrase_count_matrix : list of integer lists
                The document-keyphrase count matrix to transform to document frequencies

        Returns
        -------
        document_frequencies : np.array
            Numpy array of document frequencies for keyphrases
        """
        document_keyphrase_count_matrix = sp.csr_matrix(
            document_keyphrase_count_matrix)
        document_frequencies = np.bincount(document_keyphrase_count_matrix.indices,
                                           minlength=document_keyphrase_count_matrix.shape[1])

        return document_frequencies

    def _remove_suffixes(self, text: str, suffixes: List[str]) -> str:
        """
        Removes pre-defined suffixes from a given text string.

        Parameters
        ----------
        text : str
            Text string where suffixes should be removed.

        suffixes : list
            List of strings that should be removed from the end of the text.

        Returns
        -------
        text : Text string with removed suffixes.
        """

        for suffix in suffixes:
            if text.lower().endswith(suffix.lower()):
                return text[:-len(suffix)].strip()
        return text

    def _remove_prefixes(self, text: str, prefixes: List[str]) -> str:
        """
        Removes pre-defined prefixes from a given text string.

        Parameters
        ----------
        text : str
            Text string where prefixes should be removed.

        prefixes :  list
            List of strings that should be removed from the beginning of the text.

        Returns
        -------
        text : Text string with removed prefixes.
        """

        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                return text[len(prefix):].strip()
        return text

    def _cumulative_length_joiner(self, text_list: List[str], max_text_length: int) -> List[str]:
        """
        Joins strings from list of strings to single string until maximum char length is reached.
        Then join the next strings from list to a single string and so on.

        Parameters
        ----------
        text_list : list of strings
            List of strings to join.

        max_text_length : int
            Maximun character length of the joined strings.

        Returns
        -------
        list_of_joined_srings_with_max_length : List of joined text strings with max char length of 'max_text_length.
        """

        # triggers a parameter validation
        if isinstance(text_list, str):
            raise ValueError(
                "Iterable over raw texts expected, string object received."
            )

        # triggers a parameter validation
        if not hasattr(text_list, '__iter__'):
            raise ValueError(
                "Iterable over texts expected."
            )

        text_list_len = len(text_list) - 1
        list_of_joined_srings_with_max_length = []
        one_string = ''
        for index, text in enumerate(text_list):
            # Add the text to the substring if it doesn't make it to large
            if len(one_string) + len(text) < max_text_length:
                one_string += ' ' + text
                if index == text_list_len:
                    list_of_joined_srings_with_max_length.append(one_string)

            # Substring too large, so add to the list and reset
            else:
                list_of_joined_srings_with_max_length.append(one_string)
                one_string = text
                if index == text_list_len:
                    list_of_joined_srings_with_max_length.append(one_string)
        return list_of_joined_srings_with_max_length

    def _split_long_document(self, text: str, max_text_length: int) -> List[str]:
        """
        Split single string in list of strings with a maximum character length.

        Parameters
        ----------
        text : str
            Text string that should be split.

        max_text_length : int
            Maximun character length of the strings.

        Returns
        -------
        splitted_document : List of text strings.
        """
        # triggers a parameter validation
        if not isinstance(text, str):
            raise ValueError(
                "'text' parameter needs to be a string."
            )

        # triggers a parameter validation
        if not isinstance(max_text_length, int):
            raise ValueError(
                "'max_text_length' parameter needs to be a int"
            )

        text = text.replace("? ", "?<stop>")
        text = text.replace("! ", "!<stop>")
        if "<stop>" in text:
            splitted_document = text.split("<stop>")
            splitted_document = splitted_document[:-1]
            splitted_document = [s.strip() for s in splitted_document]
            splitted_document = [
                self._cumulative_length_joiner(text_list=doc.split(" "), max_text_length=max_text_length) if len(
                    doc) > max_text_length else [doc] for doc in splitted_document]
            return [text for doc in splitted_document for text in doc]
        else:
            splitted_document = text.split(" ")
            splitted_document = self._cumulative_length_joiner(text_list=splitted_document,
                                                               max_text_length=max_text_length)
            return splitted_document

    def _get_pos_keyphrases(self, document_list: List[str], stop_words: Union[str, List[str]], spacy_pipeline: str,
                            pos_pattern: str, spacy_exclude: List[str], custom_pos_tagger: callable,
                            lowercase: bool = True, workers: int = 1) -> List[str]:
        """
        Select keyphrases with part-of-speech tagging from a text document.
        Parameters
        ----------
        document_list : list of str
            List of text documents from which to extract the keyphrases.

        stop_words : Union[str, List[str]]
            Language of stopwords to remove from the document, e.g. 'english'.
            Supported options are `stopwords available in NLTK`_.
            Removes unwanted stopwords from keyphrases if 'stop_words' is not None.
            If given a list of custom stopwords, removes them instead.

        spacy_pipeline : str
            The name of the `spaCy pipeline`_, used to tag the parts-of-speech in the text.

        pos_pattern : str
            The `regex pattern`_ of `POS-tags`_ used to extract a sequence of POS-tagged tokens from the text.

        spacy_exclude : List[str]
            A list of `spaCy pipeline components`_ that should be excluded during the POS-tagging.
            Removing not needed pipeline components can sometimes make a big difference and improve loading and inference speed.

    custom_pos_tagger: callable
            A callable function which expects a list of strings in a 'raw_documents' parameter and returns a list of (word token, POS-tag) tuples.
            If this parameter is not None, the custom tagger function is used to tag words with parts-of-speech, while the spaCy pipeline is ignored.

        lowercase : bool, default=True
            Whether the returned keyphrases should be converted to lowercase.

        workers :int, default=1
            How many workers to use for spaCy part-of-speech tagging.
            If set to -1, use all available worker threads of the machine.
            spaCy uses the specified number of cores to tag documents with part-of-speech.
            Depending on the platform, starting many processes with multiprocessing can add a lot of overhead.
            In particular, the default start method spawn used in macOS/OS X (as of Python 3.8) and in Windows can be slow.
            Therefore, carefully consider whether this option is really necessary.

        Returns
        -------
        keyphrases : List of unique keyphrases of varying length, extracted from the text document with the defined 'pos_pattern'.
        """

        # triggers a parameter validation
        if isinstance(document_list, str):
            raise ValueError(
                "Iterable over raw text documents expected, string object received."
            )

        # triggers a parameter validation
        if not hasattr(document_list, '__iter__'):
            raise ValueError(
                "Iterable over raw text documents expected."
            )

        # triggers a parameter validation
        if not isinstance(stop_words, str) and (stop_words is not None) and (not hasattr(stop_words, '__iter__')):
            raise ValueError(
                "'stop_words' parameter needs to be a string, e.g. 'english' or 'None' or a list of strings."
            )

        # triggers a parameter validation
        if not isinstance(spacy_pipeline, str):
            raise ValueError(
                "'spacy_pipeline' parameter needs to be a spaCy pipeline string. E.g. 'en_core_web_sm'"
            )

        # triggers a parameter validation
        if not isinstance(pos_pattern, str):
            raise ValueError(
                "'pos_pattern' parameter needs to be a regex string. E.g. '<J.*>*<N.*>+'"
            )

        # triggers a parameter validation
        if ((not hasattr(spacy_exclude, '__iter__')) and (spacy_exclude is not None)) or (
                isinstance(spacy_exclude, str)):
            raise ValueError(
                "'spacy_exclude' parameter needs to be a list of 'spaCy pipeline components' strings."
            )

        # triggers a parameter validation
        if not callable(custom_pos_tagger) and (custom_pos_tagger is not None):
            raise ValueError(
                "'custom_pos_tagger' must be a callable function that gets a list of strings in a 'raw_documents' parameter and returns a list of (word, POS-tag) tuples."
            )

        # triggers a parameter validation
        if not isinstance(workers, int):
            raise ValueError(
                "'workers' parameter must be of type int."
            )

        if (workers < -1) or (workers > psutil.cpu_count(logical=True)) or (workers == 0):
            raise ValueError(
                "'workers' parameter value cannot be 0 and must be between -1 and " + str(
                    psutil.cpu_count(logical=True))
            )

        stop_words_list = []
        if isinstance(stop_words, str):
            try:
                stop_words_list = set(nltk.corpus.stopwords.words(stop_words))
            except LookupError:
                logger = logging.getLogger('KeyphraseVectorizer')
                logger.setLevel(logging.WARNING)
                sh = logging.StreamHandler()
                sh.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                logger.addHandler(sh)
                logger.setLevel(logging.DEBUG)
                logger.info(
                    'It looks like you do not have downloaded a list of stopwords yet. It is attempted to download the stopwords now.')
                nltk.download('stopwords')

                stop_words_list = set(nltk.corpus.stopwords.words(stop_words))

        elif hasattr(stop_words, '__iter__'):
            stop_words_list = stop_words

        # add spaCy POS tags for documents
        if not custom_pos_tagger:
            if not spacy_exclude:
                spacy_exclude = []
            try:
                nlp = spacy.load('zh_core_web_sm')
            except OSError:
                # set logger
                logger = logging.getLogger('KeyphraseVectorizer')
                logger.setLevel(logging.WARNING)
                sh = logging.StreamHandler()
                sh.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                logger.addHandler(sh)
                logger.setLevel(logging.DEBUG)
                logger.info(
                    'It looks like the selected spaCy pipeline is not downloaded yet. It is attempted to download the spaCy pipeline now.')
                spacy.cli.download(spacy_pipeline)
                nlp = spacy.load(spacy_pipeline,
                                 exclude=spacy_exclude)

        if workers != 1:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # split large documents in smaller chunks, so that spacy can process them without memory issues
        docs_list = []
        # set maximal character length of documents for spaCy processing
        max_doc_length = 1000000
        for document in document_list:
            if len(document) > max_doc_length:
                docs_list.extend(self._split_long_document(
                    text=document, max_text_length=max_doc_length))
            else:
                docs_list.append(document)
        document_list = docs_list
        del docs_list

        # increase max length of documents that spaCy can parse
        # (should only be done if parser and ner are not used due to memory issues)
        if not custom_pos_tagger:
            nlp.max_length = max([len(doc) for doc in document_list]) + 100

        cp = nltk.RegexpParser('CHUNK: {(' + pos_pattern + ')}')
        if not custom_pos_tagger:
            pos_tuples = []
            for tagged_doc in nlp.pipe(document_list, n_process=workers):
                pos_tuples.extend([(word.text, word.tag_)
                                  for word in tagged_doc])
        else:
            pos_tuples = custom_pos_tagger(raw_documents=document_list)

        # extract keyphrases that match the NLTK RegexpParser filter
        keyphrases = []
        # prefix_list = [stop_word + ' ' for stop_word in stop_words_list]
        # suffix_list = [' ' + stop_word for stop_word in stop_words_list]
        tree = cp.parse(pos_tuples)
        for subtree in tree.subtrees(filter=lambda tuple: tuple.label() == 'CHUNK'):
            # join candidate keyphrase from single words
            keyphrase = ' '.join([i[0] for i in subtree.leaves()])

            # convert keyphrase to lowercase
            if lowercase:
                keyphrase = keyphrase.lower()

            # remove stopword suffixes
            # keyphrase = self._remove_suffixes(keyphrase, suffix_list)

            # remove stopword prefixes
            # keyphrase = self._remove_prefixes(keyphrase, prefix_list)

            # remove whitespace from the beginning and end of keyphrases
            keyphrase = keyphrase.strip()

            # do not include single keywords that are actually stopwords
            if keyphrase.lower() not in stop_words_list:
                keyphrases.append(keyphrase)

        # remove potential empty keyphrases
        keyphrases = [keyphrase for keyphrase in keyphrases if keyphrase != '']

        return list(set(keyphrases))


class KeyphraseCountVectorizer(_KeyphraseVectorizerMixin, BaseEstimator):
    """
    KeyphraseCountVectorizer

    KeyphraseCountVectorizer converts a collection of text documents to a matrix of document-token counts.
    The tokens are keyphrases that are extracted from the text documents based on their part-of-speech tags.
    The matrix rows indicate the documents and columns indicate the unique keyphrases. Each cell represents the count.
    The part-of-speech pattern of keyphrases can be defined by the ``pos_pattern`` parameter.
    By default, keyphrases are extracted, that have 0 or more adjectives, followed by 1 or more nouns.
    A list of extracted keyphrases matching the defined part-of-speech pattern can be returned after fitting via :class:`get_feature_names_out()`.

    Attention:
        If the vectorizer is used for languages other than English, the ``spacy_pipeline`` and ``stop_words`` parameters
        must be customized accordingly.
        Additionally, the ``pos_pattern`` parameter has to be customized as the `spaCy part-of-speech tags`_  differ between languages.
        Without customizing, the words will be tagged with wrong part-of-speech tags and no stopwords will be considered.

    Parameters
    ----------
    spacy_pipeline : str, default='en_core_web_sm'
            The name of the `spaCy pipeline`_, used to tag the parts-of-speech in the text. Standard is the 'en' pipeline.

    pos_pattern :  str, default='<J.*>*<N.*>+'
        The `regex pattern`_ of `POS-tags`_ used to extract a sequence of POS-tagged tokens from the text.
        Standard is to only select keyphrases that have 0 or more adjectives, followed by 1 or more nouns.

    stop_words : Union[str, List[str]], default='english'
            Language of stopwords to remove from the document, e.g. 'english'.
            Supported options are `stopwords available in NLTK`_.
            Removes unwanted stopwords from keyphrases if 'stop_words' is not None.
            If given a list of custom stopwords, removes them instead.

    lowercase : bool, default=True
        Whether the returned keyphrases should be converted to lowercase.

    workers : int, default=1
            How many workers to use for spaCy part-of-speech tagging.
            If set to -1, use all available worker threads of the machine.
            SpaCy uses the specified number of cores to tag documents with part-of-speech.
            Depending on the platform, starting many processes with multiprocessing can add a lot of overhead.
            In particular, the default start method spawn used in macOS/OS X (as of Python 3.8) and in Windows can be slow.
            Therefore, carefully consider whether this option is really necessary.

    spacy_exclude : List[str], default=None
            A list of `spaCy pipeline components`_ that should be excluded during the POS-tagging.
            Removing not needed pipeline components can sometimes make a big difference and improve loading and inference speed.

    custom_pos_tagger: callable, default=None
            A callable function which expects a list of strings in a 'raw_documents' parameter and returns a list of (word token, POS-tag) tuples.
            If this parameter is not None, the custom tagger function is used to tag words with parts-of-speech, while the spaCy pipeline is ignored.

    max_df : int, default=None
        During fitting ignore keyphrases that have a document frequency strictly higher than the given threshold.

    min_df : int, default=None
        During fitting ignore keyphrases that have a document frequency strictly lower than the given threshold.
        This value is also called cut-off in the literature.

    binary : bool, default=False
        If True, all non zero counts are set to 1.
        This is useful for discrete probabilistic models that model binary events rather than integer counts.

    dtype : type, default=np.int64
        Type of the matrix returned by fit_transform() or transform().
    """

    def __init__(self, spacy_pipeline: str = 'en_core_web_sm', pos_pattern: str = '<J.*>*<N.*>+',
                 stop_words: Union[str, List[str]] = 'english', lowercase: bool = True, workers: int = 1,
                 spacy_exclude: List[str] = None, custom_pos_tagger: callable = None,
                 max_df: int = None, min_df: int = None, binary: bool = False, dtype: np.dtype = np.int64):

        # triggers a parameter validation
        if not isinstance(min_df, int) and min_df is not None:
            raise ValueError(
                "'min_df' parameter must be of type int"
            )
        # triggers a parameter validation
        if min_df == 0:
            raise ValueError(
                "'min_df' parameter must be > 0"
            )

        # triggers a parameter validation
        if not isinstance(max_df, int) and max_df is not None:
            raise ValueError(
                "'max_df' parameter must be of type int"
            )

        # triggers a parameter validation
        if max_df == 0:
            raise ValueError(
                "'max_df' parameter must be > 0"
            )

        # triggers a parameter validation
        if max_df and min_df and max_df <= min_df:
            raise ValueError(
                "'max_df' must be > 'min_df'"
            )

        # triggers a parameter validation
        if not isinstance(workers, int):
            raise ValueError(
                "'workers' parameter must be of type int"
            )

        if (workers < -1) or (workers > psutil.cpu_count(logical=True)) or (workers == 0):
            raise ValueError(
                "'workers' parameter value cannot be 0 and must be between -1 and " + str(
                    psutil.cpu_count(logical=True))
            )

        self.spacy_pipeline = spacy_pipeline
        self.pos_pattern = pos_pattern
        self.stop_words = stop_words
        self.lowercase = lowercase
        self.workers = workers
        self.spacy_exclude = spacy_exclude
        self.custom_pos_tagger = custom_pos_tagger
        self.max_df = max_df
        self.min_df = min_df
        self.binary = binary
        self.dtype = dtype

    def fit(self, raw_documents: List[str]) -> object:
        """
        Learn the keyphrases that match the defined part-of-speech pattern from the list of raw documents.

        Parameters
        ----------
        raw_documents : iterable
            An iterable of strings.

        Returns
        -------
        self : object
            Fitted vectorizer.
        """

        self.keyphrases = self._get_pos_keyphrases(document_list=raw_documents,
                                                   stop_words=self.stop_words,
                                                   spacy_pipeline=self.spacy_pipeline,
                                                   pos_pattern=self.pos_pattern,
                                                   lowercase=self.lowercase, workers=self.workers,
                                                   spacy_exclude=self.spacy_exclude,
                                                   custom_pos_tagger=self.custom_pos_tagger)

        # remove keyphrases that have more than 8 words, as they are probably no real keyphrases
        # additionally this prevents memory issues during transformation to a document-keyphrase matrix
        self.keyphrases = [
            keyphrase for keyphrase in self.keyphrases if len(keyphrase.split()) <= 8]

        # compute document frequencies of keyphrases
        if self.max_df or self.min_df:
            document_keyphrase_counts = CountVectorizer(vocabulary=self.keyphrases, ngram_range=(
                min([len(keyphrase.split()) for keyphrase in self.keyphrases]),
                max([len(keyphrase.split()) for keyphrase in self.keyphrases])),
                lowercase=self.lowercase, binary=self.binary,
                dtype=self.dtype).transform(
                raw_documents=raw_documents).toarray()

            document_frequencies = self._document_frequency(
                document_keyphrase_counts)

        # remove keyphrases with document frequencies < min_df and document frequencies > max_df
        if self.max_df:
            self.keyphrases = [keyphrase for index, keyphrase in enumerate(self.keyphrases) if
                               (document_frequencies[index] <= self.max_df)]
        if self.min_df:
            self.keyphrases = [keyphrase for index, keyphrase in enumerate(self.keyphrases) if
                               (document_frequencies[index] >= self.min_df)]

        # set n-gram range to zero if no keyphrases could be extracted
        if self.keyphrases:
            self.max_n_gram_length = max(
                [len(keyphrase.split()) for keyphrase in self.keyphrases])
            self.min_n_gram_length = min(
                [len(keyphrase.split()) for keyphrase in self.keyphrases])
        else:
            raise ValueError(
                "Empty keyphrases. Perhaps the documents do not contain keyphrases that match the 'pos_pattern' parameter, only contain stop words, or you set the 'min_df'/'max_df' parameters too strict.")

        return self

    def fit_transform(self, raw_documents: List[str]) -> List[List[int]]:
        """
        Learn the keyphrases that match the defined part-of-speech pattern from the list of raw documents
        and return the document-keyphrase matrix.
        This is equivalent to fit followed by transform, but more efficiently implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable of strings.

        Returns
        -------
        X : array of shape (n_samples, n_features)
            Document-keyphrase matrix.
        """

        # fit
        KeyphraseCountVectorizer.fit(self=self, raw_documents=raw_documents)

        # transform
        return CountVectorizer(vocabulary=self.keyphrases, ngram_range=(self.min_n_gram_length, self.max_n_gram_length),
                               lowercase=self.lowercase, binary=self.binary, dtype=self.dtype).fit_transform(
            raw_documents=raw_documents)

    def transform(self, raw_documents: List[str]) -> List[List[int]]:
        """
        Transform documents to document-keyphrase matrix.
        Extract token counts out of raw text documents using the keyphrases
        fitted with fit.

        Parameters
        ----------
        raw_documents : iterable
            An iterable of strings.

        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Document-keyphrase matrix.
        """

        # triggers a parameter validation
        if not hasattr(self, 'keyphrases'):
            raise NotFittedError("Keyphrases not fitted.")

        return CountVectorizer(vocabulary=self.keyphrases, ngram_range=(self.min_n_gram_length, self.max_n_gram_length),
                               lowercase=self.lowercase, binary=self.binary, dtype=self.dtype).transform(
            raw_documents=raw_documents)

    def inverse_transform(self, X: List[List[int]]) -> List[List[str]]:
        """
        Return keyphrases per document with nonzero entries in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document-keyphrase matrix.

        Returns
        -------
        X_inv : list of arrays of shape (n_samples,)
            List of arrays of keyphrase.
        """

        # triggers a parameter validation
        if not hasattr(self, 'keyphrases'):
            raise NotFittedError("Keyphrases not fitted.")

        return CountVectorizer(vocabulary=self.keyphrases, ngram_range=(self.min_n_gram_length, self.max_n_gram_length),
                               lowercase=self.lowercase, binary=self.binary, dtype=self.dtype).inverse_transform(X=X)

    @deprecated(
        "get_feature_names() is deprecated in scikit-learn 1.0 and will be removed "
        "with scikit-learn 1.2. Please use get_feature_names_out() instead."
    )
    def get_feature_names(self) -> List[str]:
        """
        Array mapping from feature integer indices to feature name.

        Returns
        -------
        feature_names : list
            A list of fitted keyphrases.
        """

        # triggers a parameter validation
        if not hasattr(self, 'keyphrases'):
            raise NotFittedError("Keyphrases not fitted.")

        # raise DeprecationWarning when function is removed from scikit-learn
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return CountVectorizer(vocabulary=self.keyphrases,
                                       ngram_range=(
                                           self.min_n_gram_length, self.max_n_gram_length),
                                       lowercase=self.lowercase, binary=self.binary,
                                       dtype=self.dtype).get_feature_names()
        except AttributeError:
            raise DeprecationWarning(
                "get_feature_names() is deprecated. Please use 'get_feature_names_out()' instead.")

    def get_feature_names_out(self) -> np.array(str):
        """
        Get fitted keyphrases for transformation.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed keyphrases.
        """

        # triggers a parameter validation
        if not hasattr(self, 'keyphrases'):
            raise NotFittedError("Keyphrases not fitted.")

        return CountVectorizer(vocabulary=self.keyphrases, ngram_range=(self.min_n_gram_length, self.max_n_gram_length),
                               lowercase=self.lowercase, binary=self.binary, dtype=self.dtype).get_feature_names_out()


class KeyphraseTfidfVectorizer(KeyphraseCountVectorizer):
    """
    KeyphraseTfidfVectorizer

    KeyphraseTfidfVectorizer converts a collection of text documents to a normalized tf or tf-idf document-token matrix.
    The tokens are keyphrases that are extracted from the text documents based on their part-of-speech tags.
    The matrix rows indicate the documents and columns indicate the unique keyphrases.
    Each cell represents the tf or tf-idf value, depending on the parameter settings.
    The part-of-speech pattern of keyphrases can be defined by the ``pos_pattern`` parameter.
    By default, keyphrases are extracted, that have 0 or more adjectives, followed by 1 or more nouns.
    A list of extracted keyphrases matching the defined part-of-speech pattern can be returned after fitting via :class:`get_feature_names_out()`.

    Attention:
        If the vectorizer is used for languages other than English, the ``spacy_pipeline`` and ``stop_words`` parameters
        must be customized accordingly.
        Additionally, the ``pos_pattern`` parameter has to be customized as the `spaCy part-of-speech tags`_  differ between languages.
        Without customizing, the words will be tagged with wrong part-of-speech tags and no stopwords will be considered.

    Tf means term-frequency while tf-idf means term-frequency times inverse document-frequency.
    This is a common term weighting scheme in information retrieval,
    that has also found good use in document classification.

    The goal of using tf-idf instead of the raw frequencies of occurrence of a token in a given document
    is to scale down the impact of tokens that occur very frequently in a given corpus and that are hence empirically less
    informative than features that occur in a small fraction of the training corpus.

    The formula that is used to compute the tf-idf for a term t of a document d in a document set is
    tf-idf(t, d) = tf(t, d) * idf(t), and the idf is computed as idf(t) = log [ n / df(t) ] + 1 (if ``smooth_idf=False``),
    where n is the total number of documents in the document set and df(t) is the document frequency of t;
    the document frequency is the number of documents in the document set that contain the term t.
    The effect of adding "1" to the idf in the equation above is that terms with zero idf, i.e., terms
    that occur in all documents in a training set, will not be entirely ignored.
    (Note that the idf formula above differs from the standard textbook
    notation that defines the idf as idf(t) = log [ n / (df(t) + 1) ]).

    If ``smooth_idf=True`` (the default), the constant "1" is added to the numerator and denominator of the idf as
    if an extra document was seen containing every term in the collection exactly once, which prevents
    zero divisions: idf(t) = log [ (1 + n) / (1 + df(t)) ] + 1.

    Furthermore, the formulas used to compute tf and idf depend on parameter settings that correspond to
    the SMART notation used in IR as follows:

    Tf is "n" (natural) by default, "l" (logarithmic) when ``sublinear_tf=True``.
    Idf is "t" when use_idf is given, "n" (none) otherwise.
    Normalization is "c" (cosine) when ``norm='l2'``, "n" (none) when ``norm=None``.

    Parameters
    ----------
    spacy_pipeline : str, default='en_core_web_sm'
            The name of the `spaCy pipeline`_, used to tag the parts-of-speech in the text. Standard is the 'en' pipeline.

    pos_pattern :  str, default='<J.*>*<N.*>+'
        The `regex pattern`_ of `POS-tags`_ used to extract a sequence of POS-tagged tokens from the text.
        Standard is to only select keyphrases that have 0 or more adjectives, followed by 1 or more nouns.

    stop_words : Union[str, List[str]], default='english'
            Language of stopwords to remove from the document, e.g. 'english'.
            Supported options are `stopwords available in NLTK`_.
            Removes unwanted stopwords from keyphrases if 'stop_words' is not None.
            If given a list of custom stopwords, removes them instead.

    lowercase : bool, default=True
        Whether the returned keyphrases should be converted to lowercase.

    workers :int, default=1
            How many workers to use for spaCy part-of-speech tagging.
            If set to -1, use all available worker threads of the machine.
            SpaCy uses the specified number of cores to tag documents with part-of-speech.
            Depending on the platform, starting many processes with multiprocessing can add a lot of overhead.
            In particular, the default start method spawn used in macOS/OS X (as of Python 3.8) and in Windows can be slow.
            Therefore, carefully consider whether this option is really necessary.

    spacy_exclude : List[str], default=None
            A list of `spaCy pipeline components`_ that should be excluded during the POS-tagging.
            Removing not needed pipeline components can sometimes make a big difference and improve loading and inference speed.

    custom_pos_tagger: callable, default=None
            A callable function which expects a list of strings in a 'raw_documents' parameter and returns a list of (word token, POS-tag) tuples.
            If this parameter is not None, the custom tagger function is used to tag words with parts-of-speech, while the spaCy pipeline is ignored.

    max_df : int, default=None
        During fitting ignore keyphrases that have a document frequency strictly higher than the given threshold.

    min_df : int, default=None
        During fitting ignore keyphrases that have a document frequency strictly lower than the given threshold.
        This value is also called cut-off in the literature.

    binary : bool, default=False
        If True, all non zero counts are set to 1.
        This is useful for discrete probabilistic models that model binary events rather than integer counts.

    dtype : type, default=np.int64
        Type of the matrix returned by fit_transform() or transform().

    norm : {'l1', 'l2'}, default='l2'
        Each output row will have unit norm, either:
        - 'l2': Sum of squares of vector elements is 1. The cosine similarity between two vectors is their dot product when l2 norm has been applied.
        - 'l1': Sum of absolute values of vector elements is 1.

    use_idf : bool, default=True
        Enable inverse-document-frequency reweighting. If False, idf(t) = 1.

    smooth_idf : bool, default=True
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.

    sublinear_tf : bool, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    """

    def __init__(self, spacy_pipeline: str = 'en_core_web_sm', pos_pattern: str = '<J.*>*<N.*>+',
                 stop_words: Union[str, List[str]] = 'english',
                 lowercase: bool = True, workers: int = 1, spacy_exclude: List[str] = None,
                 custom_pos_tagger: callable = None, max_df: int = None, min_df: int = None,
                 binary: bool = False, dtype: np.dtype = np.float64, norm: str = "l2",
                 use_idf: bool = True, smooth_idf: bool = True,
                 sublinear_tf: bool = False):

        # triggers a parameter validation
        if not isinstance(workers, int):
            raise ValueError(
                "'workers' parameter must be of type int"
            )

        if (workers < -1) or (workers > psutil.cpu_count(logical=True)) or (workers == 0):
            raise ValueError(
                "'workers' parameter value cannot be 0 and must be between -1 and " + str(
                    psutil.cpu_count(logical=True))
            )

        self.spacy_pipeline = spacy_pipeline
        self.pos_pattern = pos_pattern
        self.stop_words = stop_words
        self.lowercase = lowercase
        self.workers = workers
        self.spacy_exclude = spacy_exclude
        self.custom_pos_tagger = custom_pos_tagger
        self.max_df = max_df
        self.min_df = min_df
        self.binary = binary
        self.dtype = dtype
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

        self._tfidf = TfidfTransformer(norm=self.norm, use_idf=self.use_idf, smooth_idf=self.smooth_idf,
                                       sublinear_tf=self.sublinear_tf)

        super().__init__(spacy_pipeline=self.spacy_pipeline, pos_pattern=self.pos_pattern, stop_words=self.stop_words,
                         lowercase=self.lowercase, workers=self.workers, spacy_exclude=self.spacy_exclude,
                         custom_pos_tagger=self.custom_pos_tagger, max_df=self.max_df, min_df=self.min_df,
                         binary=self.binary, dtype=self.dtype)

    def _check_params(self):
        """
        Validate dtype parameter.
        """

        if self.dtype not in FLOAT_DTYPES:
            warnings.warn(
                "Only {} 'dtype' should be used. {} 'dtype' will "
                "be converted to np.float64.".format(FLOAT_DTYPES, self.dtype),
                UserWarning,
            )

    def fit(self, raw_documents: List[str]) -> object:
        """Learn the keyphrases that match the defined part-of-speech pattern and idf from the list of raw documents.

        Parameters
        ----------
        raw_documents : iterable
            An iterable of strings.

        Returns
        -------
        self : object
            Fitted vectorizer.
        """

        self._check_params()
        X = super().fit_transform(raw_documents)
        self._tfidf.fit(X)
        return self

    def fit_transform(self, raw_documents: List[str]) -> List[List[float]]:
        """
        Learn the keyphrases that match the defined part-of-speech pattern and idf from the list of raw documents.
        Then return document-keyphrase matrix.
        This is equivalent to fit followed by transform, but more efficiently implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable of strings.

        Returns
        -------
        X : sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-keyphrase matrix.
        """

        self._check_params()
        X = super().fit_transform(raw_documents)
        self._tfidf.fit(X)
        # X is already a transformed view of raw_documents so
        # we set copy to False
        return self._tfidf.transform(X, copy=False)

    def transform(self, raw_documents: List[str]) -> List[List[float]]:
        """
        Transform documents to document-keyphrase matrix.
        Uses the keyphrases and document frequencies (df) learned by fit (or fit_transform).

        Parameters
        ----------
        raw_documents : iterable
            An iterable of strings.

        Returns
        -------
        X : sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-keyphrase matrix.
        """

        # triggers a parameter validation
        if not hasattr(self, 'keyphrases'):
            raise NotFittedError("Keyphrases not fitted.")

        X = super().transform(raw_documents)
        return self._tfidf.transform(X, copy=False)
