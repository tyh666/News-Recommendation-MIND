import random
import re
import os
import json
import pickle
import math
import torch
import logging
import transformers
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s (%(name)s) %(message)s")
transformers.logging.set_verbosity_error()

logger = logging.getLogger(__name__)


# stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you"re", "you"ve", "you"ll", "you"d", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "she"s", "her", "hers", "herself", "it", "it"s", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "that"ll", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "don"t", "should", "should"ve", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren"t", "couldn", "couldn"t", "didn", "didn"t", "doesn", "doesn"t", "hadn", "hadn"t", "hasn", "hasn"t", "haven", "haven"t", "isn", "isn"t", "ma", "mightn", "mightn"t", "mustn", "mustn"t", "needn", "needn"t", "shan", "shan"t", "shouldn", "shouldn"t", "wasn", "wasn"t", "weren", "weren"t", "won", "won"t", "wouldn", "wouldn"t"]

def tokenize(sent):
    """ Split sentence into words
    Args:
        sent (str): Input sentence

    Return:
        list: word list
    """
    pat = re.compile(r"[-\w_]+|[.,!?;|]")

    return [x for x in pat.findall(sent.lower())]


def convert_tokens_to_words_bert(tokens):
    """
    transform the tokens output by tokenizer to words (connecting subwords)

    Args:
        tokens: list of tokens

    Returns:
        words: list of words, without [PAD]
    """
    words = []
    for tok in tokens:
        if tok.startswith("##"):
            words[-1] += tok[2:]
        elif tok == "[PAD]":
            break
        else:
            words.append(tok)

    return words


def _group_lists(impr_indexes, *associated_lists):
        """
            group lists by impr_index
        Args:
            associated_lists: list of lists, where list[i] is associated with the impr_indexes[i]

        Returns:
            Iterable: grouped labels (if inputted) and preds
        """
        list_num = len(associated_lists)
        dicts = [defaultdict(list) for i in range(list_num)]

        for x in zip(impr_indexes, *associated_lists):
            key = x[0]
            values = x[1:]
            for i in range(list_num):
                dicts[i][key].extend(values[i])

        grouped_lists = [list(d.values()) for d in dicts]

        return grouped_lists


def newsample(news, ratio):
    """ Sample ratio samples from news list.
    If length of news is less than ratio, pad zeros.

    Args:
        news (list): input news list
        ratio (int): sample number

    Returns:
        list: output of sample list.
        int: count of paddings
    """
    if ratio > len(news):
        return news + [0] * (ratio - len(news)), len(news)
    else:
        return random.sample(news, ratio), ratio


def construct_sequential_behaviors(path):
    """ construct sequential behavior logs and save to the input directory

    Args:
        path: the directory of MINDscale (with slash)

    """
    behaviors = defaultdict(list)
    train_path = path + "behaviors.tsv"

    with open(train_path, "r", encoding="utf-8") as rd:
        for idx in rd:
            impr_index, uid, time, history, impr = idx.strip("\n").split("\t")
            # important to subtract 1 because all list related to behaviors start from 0

            behaviors[uid].append([impr_index, uid, time, history, impr])

    for k,v in behaviors.items():
        behaviors[k] = sorted(v,key=lambda x: datetime.strptime(x[2], "%m/%d/%Y %X %p"))

    for k,v in behaviors.items():
        tmp = []
        for i,behav in enumerate(v):
            impr = behav[-1].split()
            impr_news = [i.split("-")[0] for i in impr]
            labels = [i.split("-")[1] for i in impr]

            for news,label in zip(impr_news, labels):
                if(label == "1"):
                    tmp.append(news)

            if(i > 0 and tmp):
                behav[3] = behav[3] + " " +" ".join(tmp)

    with open(path + "behaviors_sequential.tsv","w",encoding="utf-8") as f:
        for k,v in behaviors.items():
            for behav in v:
                f.write("\t".join(behav) + "\n")


def statistic_MIND(config):
        """
            analyse over MIND
        """
        mind_path = config.path + "MIND"

        avg_title_length = 0
        avg_abstract_length = 0
        avg_his_length = 0
        avg_imp_length = 0
        cnt_his_lg_50 = 0
        cnt_his_eq_0 = 0
        cnt_imp_multi = 0

        news_file = mind_path + \
            "/MIND{}_{}/news.tsv".format(config.scale, config.mode)

        behavior_file = mind_path + \
            "/MIND{}_{}/behaviors.tsv".format(config.scale, config.mode)

        with open(news_file, "r", encoding="utf-8") as rd:
            count = 0
            for idx in rd:
                nid, vert, subvert, title, ab, url, _, _ = idx.strip(
                    "\n").split("\t")
                avg_title_length += len(title.split(" "))
                avg_abstract_length += len(ab.split(" "))
                count += 1
        avg_title_length = avg_title_length/count
        avg_abstract_length = avg_abstract_length/count

        with open(behavior_file, "r", encoding="utf-8") as rd:
            count = 0
            for idx in rd:
                uid, time, history, impr = idx.strip("\n").split("\t")[-4:]
                his = history.split(" ")
                imp = impr.split(" ")
                if len(his) > 50:
                    cnt_his_lg_50 += 1
                if len(imp) > 50:
                    cnt_imp_multi += 1
                if not his[0]:
                    cnt_his_eq_0 += 1
                avg_his_length += len(his)
                avg_imp_length += len(imp)
                count += 1
        avg_his_length = avg_his_length/count
        avg_imp_length = avg_imp_length/count

        print("avg_title_length:{}\n avg_abstract_length:{}\n avg_his_length:{}\n avg_impr_length:{}\n cnt_his_lg_50:{}\n cnt_his_eq_0:{}\n cnt_imp_multi:{}".format(
            avg_title_length, avg_abstract_length, avg_his_length, avg_imp_length, cnt_his_lg_50, cnt_his_eq_0, cnt_imp_multi))



def getId2idx(file):
    """
        get Id2idx dictionary from json file
    """
    g = open(file, "r", encoding="utf-8")
    dic = json.load(g)
    g.close()
    return dic


def collate_recall(data):
    result = defaultdict(list)
    for d in data:
        for k, v in d.items():
            result[k].append(v)
    for k, v in result.items():
        if "cdd" not in k:
            result[k] = torch.from_numpy(np.asarray(v))
        else:
            result[k] = v
    return dict(result)



def construct_inverted_index(corpus, score_func):
    """ construct inverted index on corpus according to score_func
    key: token
    value: list of tuples, the first element in a tuple is document index in the corpus; the second one is its score
    """
    logger.info("initializing inverted index of scoring function {}...".format(score_func.name))
    inverted_index = defaultdict(list)
    for i,document in enumerate(corpus):
        token_set = set()
        # strip [CLS]
        for token in document:
            if token not in token_set and token not in [0, 101, 102]:
                inverted_index[token].append([i, score_func(token, i)])
                token_set.add(token)

    inverted_array = np.zeros((30522, 100, 2))
    padding = len(corpus)

    # make sure that absent token doesn't recall news
    inverted_array[:, :, 0] = padding

    for k,v in inverted_index.items():
        v = sorted(v, key=lambda x: x[1], reverse=True)[:100] + [[padding,0.]] * (100 - len(v))
        inverted_array[k] = np.asarray(v)

    inverted_array = torch.from_numpy(inverted_array)
    torch.save(inverted_array, "data/recall/inverted_index_bm25.pt")
    return inverted_array

    # inverted_index = [[]] * token_num
    # for i,document in enumerate(corpus):
    #     token_set = set()
    #     # strip [CLS]
    #     for token in document:
    #         if token not in token_set:
    #             inverted_index[token].append([i, score_func(token, i)])
    #             token_set.add(token)

    # for i,v in enumerate(inverted_index):
    #     v = np.asarray(v)
    #     # sort by the last column i.e. score
    #     inverted_index[i] = v[v[:, 1].argsort()[::-1]]

    # inverted_index = np.asarray(inverted_index, dtype=object)
    # return inverted_index



class Partition_Sampler():
    def __init__(self, dataset, num_replicas, rank) -> None:
        super().__init__()
        len_per_worker, extra_len = divmod(len(dataset), num_replicas)
        self.start = len_per_worker * rank
        self.end = self.start + len_per_worker + extra_len * (rank + 1 == num_replicas)
        # store the partition points
        self.partition_points = list(range(0, len(dataset) + 1, len_per_worker))

    def __iter__(self):
        start = self.start
        end = self.end

        return iter(range(start, end, 1))

    def __len__(self):
        return self.end - self.start



class BM25_token(object):
    """
    compute bm25 score of every term in the corpus
    """
    def __init__(self, documents, k=2, epsilon=0.5):
        """
        Args:
            documents: list of tokens
        """
        self.name = "bm25-token"

        self.k = k
        self.epsilon = epsilon
        # initialize tf and idf
        self._build_tf_idf(documents)
        logger.info("initializing token-level BM25...")

    def _build_tf_idf(self, documents):
        """
        build term frequencies (how many times a term occurs in one news) and document frequencies (how many documents contains a term)
        """
        doc_count = len(documents)

        tfs = []
        df = defaultdict(int)
        for document in documents:
            tf = defaultdict(int)
            # strip [CLS]
            for token in document[1:]:
                # term frequency in this document
                tf[token] += 1
                # global document frequency
                df[token] += 1
            tfs.append(tf)

        self.tfs = tfs

        idf = defaultdict(float)
        for word,freq in df.items():
            idf[word] = math.log((doc_count - freq + 0.5 ) / (freq + 0.5) + 1)
        # inverse document frequency
        self.idf = idf

    def __call__(self, token, doc_idx):
        """
        compute bm25 score of the given token

        Args:
            token: int

        Returns:
            bm25 score of the token
        """
        tf = self.tfs[doc_idx][token]
        score = (self.idf[token] * tf * (self.k + 1)) / (tf + self.k)
        return score


class BM25(object):
    """
    compute bm25 score on the entire corpus, instead of the one limited by signal_length
    """
    def __init__(self, k=2, epsilon=0.5):
        self.k = k
        self.epsilon = epsilon

    def _build_tf_idf(self, documents):
        """
        build term frequencies (how many times a term occurs in one news) and document frequencies (how many documents contains a term)
        """
        doc_count = len(documents)

        tfs = []
        df = defaultdict(int)
        for document in documents:
            tf = defaultdict(int)
            words = re.sub("[.&*()+=/\<>,!?;:~`@#$%^]", '', document).split()
            for word in words:
                tf[word] += 1
                df[word] += 1
            tfs.append(tf)

        self.tfs = tfs

        idf = defaultdict(float)
        for word,freq in df.items():
            idf[word] = math.log((doc_count - freq + 0.5 ) / (freq + 0.5) + 1)

        self.idf = idf

    def __call__(self, documents):
        """
        compute bm25 score of each word in each document and sort the words by accordingly
        with b=0, totally ignoring the effect of document length

        note that [CLS] always stays at the head of the output

        Args:
            documents: list of lists

        Returns:
            sorted tokens
        """
        logger.info("computing BM25 scores...")
        self._build_tf_idf(documents)

        bm25_scores = []
        for tf in self.tfs:
            score = defaultdict(float)
            for word, freq in tf.items():
                score[word] = (self.idf[word] * freq * (self.k + 1)) / (freq + self.k)

            bm25_scores.append(dict(sorted(score.items(), key=lambda item: item[1], reverse=True)))

        sorted_documents = []
        for i, bm25 in enumerate(bm25_scores):
            # [CLS] token in the front
            if not i:
                # the first article is empty
                sorted_documents.append("")
            else:
                sorted_documents.append(" ".join(list(bm25.keys())))

        return sorted_documents


class DoNothing(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, documents):
        return documents


class DeDuplicate(object):
    """
    mask duplicated terms in one document by attention masks
    """
    def __init__(self, manager) -> None:
        super().__init__()
        self.max_length = manager.signal_length

        if not manager.no_rm_punc:
            punc_map = {
                'bert':set([102,999,1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1066,1529,1996]),
                "deberta":set([2,10975,4,947,3226,1640,43,2744,5214,73,37457,41552,15698,6,328,116,131,35,34437,12905,1039,10431,1629,207,35227,742])
            }
            self.punctuations = punc_map[manager.embedding]
        else:
            self.punctuations = ''

    def __call__(self, documents, attn_masks):
        """
            1. set the attention mask of duplicated tokens to 0
            2. only keep the first max_length tokens per article
        """
        # do not modify the orginal attention mask
        attn_masks = attn_masks.copy()[:, :self.max_length]

        # logger.info("deduplicating...")
        for i, document in enumerate(documents):
            tokens = set()
            duplicated = []
            for j, token in enumerate(document):
                if token in tokens or token in self.punctuations:
                    # if the term duplicates
                    duplicated.append(j)
                else:
                    tokens.add(token)

            # in case the non-duplicate terms are more than k, then mask duplicated terms
            attn_masks[i, duplicated] = 0

        return documents, attn_masks


class CountFreq(object):
    """
    generate token count pairs
    """
    def __init__(self, manager) -> None:
        super().__init__()
        self.max_length = manager.signal_length
        self.position = manager.save_pos

    def __call__(self, documents, attn_masks):
        """
        count unique tokens in the first max_length per article

        Returns:
            documents: list of list of tuples, [[(word1 : freq1), ...] ...]
            positions: [[]]
        """
        # logger.info("reducing to Bag-of-Words...")
        token_counts = []
        attn_masks = []

        documents = documents[:, :self.max_length]

        for document in documents:
            token_count = defaultdict(int)
            for token in document:
                if token == 0:
                    break
                token_count[token] += 1

            pad_length = self.max_length - len(token_count)
            token_count_tuples = list(token_count.items()) + [[0, 0]] * pad_length
            attn_mask = [1] * len(token_count) + [0] * pad_length

            token_counts.append(token_count_tuples)
            attn_masks.append(attn_mask)

        attn_masks = np.asarray(attn_masks)

        return np.asarray(token_counts), attn_masks


class Truncate(object):
    def __init__(self, manager) -> None:
        super().__init__()
        self.max_length = manager.signal_length

    def __call__(self, documents, attn_masks):
        """
            only keep the first max_length tokens per article
        """
        return documents[:, :self.max_length], attn_masks[:, :self.max_length]


class DoNothing(object):
    def __init__(self):
        super().__init__()

    def __call__(self, documents, attn_masks):
        return documents, attn_masks