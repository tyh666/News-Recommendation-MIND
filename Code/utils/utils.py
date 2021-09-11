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
import torch.distributed as dist
from collections import defaultdict
from datetime import datetime
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

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


def convert_tokens_to_words_deberta(tokens):
    """
    transform the tokens output by tokenizer to words (connecting subwords)

    Args:
        tokens: list of tokens

    Returns:
        words: list of words, without [PAD]
    """
    words = []
    for i,tok in enumerate(tokens):
        # the beginning of a word
        if i == 0:
            words.append(tok)
        elif tok == "[PAD]":
            break
        elif tok.startswith("Ġ"):
            words.append(tok[1:])
        else:
            words[-1] += tok

    return words


def convert_tokens_to_words_deberta_punctuation(tokens):
    """
    transform the tokens output by tokenizer to words (connecting subwords)

    Args:
        tokens: list of tokens

    Returns:
        words: list of words, without [PAD]
    """
    words = []
    for i,tok in enumerate(tokens):
        # the beginning of a word
        if i == 0:
            words.append(tok)
        elif tok == "[PAD]":
            break
        elif tok.startswith("Ġ"):
            words.append(tok[1:])
        elif tok in r"[.&*()+=/\<>,!?;:~`@#$%^]":
            words.append(tok)
        else:
            words[-1] += tok

    return words


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


def tailor_data(tsvFile, num):
    """ tailor num rows of tsvFile to create demo data file

    Args:
        tsvFile: str of data path
    Returns:
        create tailored data file
    """
    pattern = re.search("(.*)MIND(.*)_(.*)/(.*).tsv", tsvFile)

    directory = pattern.group(1)
    mode = pattern.group(3)
    behavior_file = pattern.group(4)

    if not os.path.exists(directory + "MINDdemo" + "_{}/".format(mode)):
        os.makedirs(directory + "MINDdemo" + "_{}/".format(mode))

    behavior_file = directory + "MINDdemo" + \
        "_{}/".format(mode) + behavior_file + ".tsv"

    f = open(behavior_file, "w", encoding="utf-8")
    count = 0
    with open(tsvFile, "r", encoding="utf-8") as g:
        for line in g:
            if count >= num:
                f.close()
                break
            f.write(line)
            count += 1
    news_file = re.sub("behaviors", "news", tsvFile)
    news_file_new = re.sub("behaviors", "news", behavior_file)

    os.system("cp {} {}".format(news_file, news_file_new))
    logger.info("tailored {} behaviors to {}, copied news file also".format(
        num, behavior_file))
    return


def expand_data():
    """ Beta
    """
    a = pd.read_table(r"D:\Data\MIND\MINDlarge_train\behaviors.tsv",
                      index_col=0, names=["a", "b", "c", "d", "e"], quoting=3)
    b = pd.read_table(r"D:\Data\MIND\MINDlarge_dev\behaviors.tsv",
                      index_col=0, names=["a", "b", "c", "d", "e"], quoting=3)
    c = pd.concat([a, b]).drop_duplicates().reset_index(inplace=True)
    c = c[["b", "c", "d", "e"]]

    c.to_csv(r"D:\Data\MIND\MINDlarge_whole\behaviors.tsv",
             index=True, sep="\t", header=False)


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


def getId2idx(file):
    """
        get Id2idx dictionary from json file
    """
    g = open(file, "r", encoding="utf-8")
    dic = json.load(g)
    g.close()
    return dic


def getVocab(file):
    """
        get Vocabulary from pkl file
    """
    g = open(file, "rb")
    dic = pickle.load(g)
    g.close()
    return dic


def my_collate(data):
    result = defaultdict(list)
    for d in data:
        for k, v in d.items():
            result[k].append(v)
    for k, v in result.items():
        result[k] = torch.from_numpy(np.asarray(v))
    return dict(result)


def manual_seed(seed):
    # os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

def prepare(config):
    from .MIND import MIND,MIND_news,MIND_impr
    """ prepare dataloader and several paths

    Args:
        config(dict): hyper parameters

    Returns:
        vocab
        loaders(list of dataloaders): 0-loader_train/test/dev, 1-loader_dev, 2-loader_validate
    """
    if config.rank in [-1, 0]:
        logger.info("Hyper Parameters are {}".format(config))
        logger.info("preparing dataset...")

    if config.seeds:
        manual_seed(config.seeds)

    mind_path = config.path + "MIND"
    shuffle = config.shuffle
    pin_memory = config.pin_memory
    num_workers = config.num_workers

    # if impr:
    #     # FIXME: if config.bert
    #     news_file_dev = mind_path+"/MIND"+config.scale+"_dev/news.tsv"
    #     behavior_file_dev = mind_path+"/MIND"+config.scale+"_dev/behaviors.tsv"

    #     dataset_dev = MIND_impr(config=config, news_file=news_file_dev,
    #                         behaviors_file=behavior_file_dev)
    #     loader_dev = DataLoader(dataset_dev, batch_size=1, pin_memory=pin_memory,
    #                             num_workers=num_workers, drop_last=False)
    #     vocab = dataset_dev.vocab
    #     if not config.bert:
    #         embedding = GloVe(dim=300, cache=vec_cache_path)
    #         vocab.load_vectors(embedding)

    #     return vocab, [loader_dev]

    # if news:
    #     news_file_train = mind_path + \
    #         "/MIND{}_train/news.tsv".format(config.scale)
    #     news_file_dev = mind_path+"/MIND{}_dev/news.tsv".format(config.scale)
    #     # FIXME: if config.bert
    #     dataset_train = MIND_news(config, news_file_train)
    #     loader_news_train = DataLoader(
    #         dataset_train, batch_size=config.batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False)

    #     dataset_dev = MIND_news(config, news_file_dev)
    #     loader_news_dev = DataLoader(
    #         dataset_dev, batch_size=1, pin_memory=pin_memory, num_workers=num_workers, drop_last=False)

    #     vocab = getVocab("data/dictionaries/vocab.pkl")
    #     embedding = GloVe(dim=300, cache=vec_cache_path)
    #     vocab.load_vectors(embedding)

    #     if config.scale == "large":
    #         news_file_test = mind_path + \
    #             "/MIND{}_test/news.tsv".format(config.scale)
    #         dataset_test = MIND_news(config, news_file_test)
    #         loader_news_test = DataLoader(
    #             dataset_test, batch_size=config.batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False)

    #         return vocab, [loader_news_train, loader_news_dev, loader_news_test]

    #     return vocab, [loader_news_train, loader_news_dev]


    if config.mode in ["train", "tune"]:
        news_file_train = mind_path+"/MIND"+config.scale+"_train/news.tsv"
        behavior_file_train = mind_path+"/MIND" + \
            config.scale+"_train/behaviors.tsv"
        news_file_dev = mind_path+"/MIND"+config.scale+"_dev/news.tsv"
        behavior_file_dev = mind_path+"/MIND"+config.scale+"_dev/behaviors.tsv"

        dataset_train = MIND(config=config, news_file=news_file_train,
                        behaviors_file=behavior_file_train)
        dataset_dev = MIND(config=config, news_file=news_file_dev,
                        behaviors_file=behavior_file_dev)

        # FIXME: multi view dataset

        if config.world_size > 0:
            sampler_train = DistributedSampler(dataset_train, num_replicas=config.world_size, rank=config.rank, shuffle=shuffle)
            sampler_dev = Partition_Sampler(dataset_dev, num_replicas=config.world_size, rank=config.rank)
        else:
            sampler_train = None
            sampler_dev = None
        loader_train = DataLoader(dataset_train, batch_size=config.batch_size, pin_memory=pin_memory,
                                num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler_train)
        loader_dev = DataLoader(dataset_dev, batch_size=1, pin_memory=pin_memory,
                                num_workers=num_workers, drop_last=False, sampler=sampler_dev)

        return loader_train, loader_dev

    elif config.mode in ["dev", "inspect"]:
        news_file_dev = mind_path+"/MIND"+config.scale+"_dev/news.tsv"
        behavior_file_dev = mind_path+"/MIND"+config.scale+"_dev/behaviors.tsv"

        dataset_dev = MIND(config=config, news_file=news_file_dev,
                            behaviors_file=behavior_file_dev)

        if config.world_size > 0:
            sampler_dev = Partition_Sampler(dataset_dev, num_replicas=config.world_size, rank=config.rank)
        else:
            sampler_dev = None
        loader_dev = DataLoader(dataset_dev, batch_size=1, pin_memory=pin_memory,
                                num_workers=num_workers, drop_last=False, sampler=sampler_dev)

        return loader_dev,

    elif config.mode == "test":
        news_file_test = mind_path+"/MIND"+config.scale+"_test/news.tsv"
        behavior_file_test = mind_path+"/MIND"+config.scale+"_test/behaviors.tsv"

        dataset_test = MIND(config, news_file_test, behavior_file_test)

        # FIXME distributed test
        if config.world_size > 0:
            sampler_test = Partition_Sampler(dataset_test, num_replicas=config.world_size, rank=config.rank)
        else:
            sampler_test = None
        loader_test = DataLoader(dataset_test, batch_size=1, pin_memory=pin_memory,
                                 num_workers=num_workers, drop_last=False, sampler=sampler_test)

        return loader_test,


def analyse(config):
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


def setup(rank, manager):
    """
    set up distributed training and fix seeds
    """
    if manager.world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=manager.world_size)

        os.environ["TOKENIZERS_PARALLELISM"] = "True"
        # manager.rank will be invoked in creating DistributedSampler
        manager.rank = rank
        # manager.device will be invoked in the model
        manager.device = rank

    else:
        # one-gpu
        manager.rank = -1

    if rank != "cpu":
        torch.cuda.set_device(rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)


def cleanup():
    """
    shut down the distributed training process
    """
    dist.destroy_process_group()


class Partition_Sampler():
    def __init__(self, dataset, num_replicas, rank) -> None:
        super().__init__()
        len_per_worker, extra_len = divmod(len(dataset), num_replicas)
        self.start = len_per_worker * rank
        self.end = self.start + len_per_worker + extra_len * (rank + 1 == num_replicas)

    def __iter__(self):
        start = self.start
        end = self.end

        return iter(range(start, end, 1))

    def __len__(self):
        return self.end - self.start


class BM25(object):
    """
    compute bm25 score
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
            words = document.split()
            # ignore [CLS]
            for word in words[1:]:
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
                sorted_documents.append(" ".join(["[CLS]"] + list(bm25.keys())))

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
    def __init__(self, config) -> None:
        super().__init__()
        self.max_length = config.signal_length

        if not config.no_rm_punc:
            punc_map = {
                'bert':set([1031,1012,1004,1008,1006,1007,1009,1027,1013,1032,1026,1028,1010,999,1029,1025,1024,1066,1036,1030,1001,1002,1003,1034,1033]),
                "deberta":set([10975,4,947,3226,1640,43,2744,5214,73,37457,41552,15698,6,328,116,131,35,34437,12905,1039,10431,1629,207,35227,742])
            }
            self.punctuations = punc_map[config.embedding]
        else:
            self.punctuations = ''

    def __call__(self, documents, attn_masks):
        """
            1. set the attention mask of duplicated tokens to 0
            2. only keep the first max_length tokens per article
        """
        # do not modify the orginal attention mask
        documents = documents[:, :self.max_length]
        attn_masks = attn_masks.copy()[:, :self.max_length]

        logger.info("deduplicating...")
        for i, document in enumerate(documents):
            tokens = set()
            duplicated = []
            # ignore [CLS]
            for j, token in enumerate(document[1:]):
                if token in tokens or token in self.punctuations:
                    # if the term duplicates
                    # [CLS] token always stands ahead
                    duplicated.append(j + 1)
                else:
                    tokens.add(token)

            # in case the non-duplicate terms are more than k, then mask duplicated terms
            attn_masks[i, duplicated] = 0

        return documents, attn_masks


class CountFreq(object):
    """
    generate token count pairs
    """
    def __init__(self, config) -> None:
        super().__init__()
        self.max_length = config.signal_length
        self.position = config.save_pos

    def __call__(self, documents, attn_masks):
        """
        count unique tokens in the first max_length per article

        Returns:
            documents: list of list of tuples, [[(word1 : freq1), ...] ...]
            positions: [[]]
        """
        logger.info("reducing to Bag-of-Words...")
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
    def __init__(self, config) -> None:
        super().__init__()
        self.max_length = config.signal_length

    def __call__(self, documents, attn_masks):
        """
            only keep the first max_length tokens per article
        """
        return documents[:, :self.max_length], attn_masks[:, :self.max_length]