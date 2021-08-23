import random
import re
import os
import json
import pickle
import math
import torch
import argparse
import logging
import transformers
import pandas as pd
import numpy as np
import torch.distributed as dist
# import torch.multiprocessing as mp
from collections import defaultdict
from datetime import datetime
# from itertools import product
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.Manager import Manager

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s (%(name)s) %(message)s")
transformers.logging.set_verbosity_error()

logger = logging.getLogger(__name__)

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

def tokenize(sent):
    """ Split sentence into words
    Args:
        sent (str): Input sentence

    Return:
        list: word list
    """
    pat = re.compile(r"[-\w_]+|[.,!?;|]")

    return [x for x in pat.findall(sent.lower()) if x not in stopwords]


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
        return news + [0] * (ratio - len(news)), ratio-len(news)
    else:
        return random.sample(news, ratio), 0


def news_token_generator(news_file_list, tokenizer, attrs):
    """ merge and deduplicate training news and testing news then iterate, collect attrs into a single sentence and generate it

    Args:
        tokenizer: torchtext.data.utils.tokenizer
        attrs: list of attrs to be collected and yielded
    Returns:
        a generator over attrs in news
    """
    news_df_list = []
    for f in news_file_list:
        news_df_list.append(pd.read_table(f, index_col=None, names=[
                            "newsID", "category", "subcategory", "title", "abstract", "url", "entity_title", "entity_abstract"], quoting=3))

    news_df = pd.concat(news_df_list).drop_duplicates().dropna()
    news_iterator = news_df.iterrows()

    for _, i in news_iterator:
        content = []
        for attr in attrs:
            content.append(i[attr])

        yield tokenizer(" ".join(content))


def construct_vocab(news_file_list, attrs):
    """
        Build field using torchtext for tokenization

    Returns:
        torchtext.vocabulary
    """
    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(
        news_token_generator(news_file_list, tokenizer, attrs))

    # adjustments for torchtext >= 0.10.0
    # vocab.insert_token('[PAD]', 0)
    # vocab.insert_token('[UNK]', 0)
    # vocab.set_default_index(0)

    output = open(
        "data/dictionaries/vocab_{}.pkl".format(",".join(attrs)), "wb")
    pickle.dump(vocab, output)
    output.close()


def construct_nid2idx(news_file, scale, mode):
    """
        Construct news to newsID dictionary, index starting from 1
    """
    nid2index = {}

    news_df = pd.read_table(news_file, index_col=None, names=[
                            "newsID", "category", "subcategory", "title", "abstract", "url", "entity_title", "entity_abstract"], quoting=3)

    for v in news_df["newsID"]:
        if v in nid2index:
            continue
        nid2index[v] = len(nid2index) + 1

    h = open("data/dictionaries/nid2idx_{}_{}.json".format(scale, mode), "w")
    json.dump(nid2index, h, ensure_ascii=False)
    h.close()


def construct_uid2idx(behavior_file_list, scale):
    """
        Construct user to userID dictionary, index starting from 1
    """
    uid2index = {}

    user_df_list = []
    for f in behavior_file_list:
        user_df_list.append(pd.read_table(f, index_col=None, names=[
                            "imprID", "uid", "time", "hisstory", "abstract", "impression"], quoting=3))

    user_df = pd.concat(user_df_list).drop_duplicates()

    for v in user_df["uid"]:
        if v in uid2index:
            continue
        uid2index[v] = len(uid2index) + 1

    h = open("data/dictionaries/uid2idx_{}.json".format(scale), "w")
    json.dump(uid2index, h, ensure_ascii=False)
    h.close()


def construct_basic_dict(attrs=['title','abstract','category','subcategory'], path="../../../Data/MIND"):
    """
        construct basic dictionary
    """
    news_file_list = [path + "/MINDlarge_train/news.tsv", path +
                       "/MINDlarge_dev/news.tsv", path + "/MINDlarge_test/news.tsv"]
    construct_vocab(news_file_list, attrs)

    for scale in ["demo", "small", "large"]:
        news_file_list = [path + "/MIND{}_train/news.tsv".format(
            scale), path + "/MIND{}_dev/news.tsv".format(scale), path + "/MIND{}_test/news.tsv".format(scale)]
        behavior_file_list = [path + "/MIND{}_train/behaviors.tsv".format(
            scale), path + "/MIND{}_dev/behaviors.tsv".format(scale), path + "/MIND{}_test/behaviors.tsv".format(scale)]

        if scale == "large":
            news_file_train = news_file_list[0]
            news_file_dev = news_file_list[1]
            news_file_test = news_file_list[2]

            construct_nid2idx(news_file_train, scale, "train")
            construct_nid2idx(news_file_dev, scale, "dev")
            construct_nid2idx(news_file_test, scale, "test")

            construct_uid2idx(behavior_file_list, scale)

        else:
            news_file_list = news_file_list[0:2]

            news_file_train = news_file_list[0]
            news_file_dev = news_file_list[1]

            construct_nid2idx(news_file_train, scale, "train")
            construct_nid2idx(news_file_dev, scale, "dev")

            behavior_file_list = behavior_file_list[0:2]
            construct_uid2idx(behavior_file_list, scale)


def construct_vert_onehot():
    import pandas as pd
    path = "/home/peitian_zhang/Data/MIND"
    news_file_list = [path + "/MINDlarge_train/news.tsv", path +
                        "/MINDlarge_dev/news.tsv", path + "/MINDlarge_test/news.tsv"]
    news_df_list = []
    for f in news_file_list:
        news_df_list.append(pd.read_table(f, index_col=None, names=["newsID", "category", "subcategory", "title", "abstract", "url", "entity_title", "entity_abstract"], quoting=3))

    news_df = pd.concat(news_df_list).drop_duplicates()

    vert = news_df["category"].unique()
    subvert = news_df["subcategory"].unique()
    vocab = getVocab("data/dictionaries/vocab_whole.pkl")
    vert2idx = {
        vocab[v]:i for i,v in enumerate(vert)
    }
    subvert2idx = {
        vocab[v]:i for i,v in enumerate(subvert)
    }
    vert2onehot = {}
    for k,v in vert2idx.items():
        a = np.zeros((len(vert2idx)))
        index = np.asarray([v])
        a[index] = 1
        vert2onehot[int(k)] = a.tolist()
    vert2onehot[1] = [0]*len(next(iter(vert2onehot.values())))

    subvert2onehot = {}
    for k,v in subvert2idx.items():
        a = np.zeros((len(subvert2idx)))
        index = np.asarray([v])
        a[index] = 1
        subvert2onehot[int(k)] = a.tolist()
    subvert2onehot[1] = [0]*len(next(iter(subvert2onehot.values())))

    json.dump(vert2onehot, open("data/dictionaries/vert2onehot.json","w"),ensure_ascii=False)
    json.dump(subvert2onehot, open("data/dictionaries/subvert2onehot.json","w"),ensure_ascii=False)


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
    train_path = path + 'behaviors.tsv'

    with open(train_path, "r", encoding='utf-8') as rd:
        for idx in rd:
            impr_index, uid, time, history, impr = idx.strip("\n").split('\t')
            # important to subtract 1 because all list related to behaviors start from 0

            behaviors[uid].append([impr_index, uid, time, history, impr])

    for k,v in behaviors.items():
        behaviors[k] = sorted(v,key=lambda x: datetime.strptime(x[2], '%m/%d/%Y %X %p'))

    for k,v in behaviors.items():
        tmp = []
        for i,behav in enumerate(v):
            impr = behav[-1].split()
            impr_news = [i.split("-")[0] for i in impr]
            labels = [i.split("-")[1] for i in impr]

            for news,label in zip(impr_news, labels):
                if(label == '1'):
                    tmp.append(news)

            if(i > 0 and tmp):
                behav[3] = behav[3] + ' ' +' '.join(tmp)

    with open(path + 'behaviors_sequential.tsv','w',encoding='utf-8') as f:
        for k,v in behaviors.items():
            for behav in v:
                f.write('\t'.join(behav) + '\n')


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
    excluded = ["impression_index"]
    result = defaultdict(list)
    for d in data:
        for k, v in d.items():
            result[k].append(v)
    for k, v in result.items():
        if k not in excluded:
            result[k] = torch.from_numpy(np.asarray(v))

        else:
            continue
    return dict(result)


def info(config):
    return "\n".join(["{}:{}".format(k,v) for k,v in vars(config).items() if not k.startswith('__')])


def load_manager():
    """
        customize hyper parameters in command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scale", dest="scale", help="data scale",
                        choices=["demo", "small", "large", "whole"], required=True)
    parser.add_argument("-m", "--mode", dest="mode", help="train or test",
                        choices=["train", "dev", "test", "tune", "encode"], default="train")
    parser.add_argument("-e", "--epochs", dest="epochs",
                        help="epochs to train the model", type=int, default=10)
    parser.add_argument("-d","--device", dest="device",
                        help="device to run on, -1 means cpu", choices=[i for i in range(-1,10)], type=int, default=0)
    parser.add_argument("-p", "--path", dest="path", type=str, default="../../../Data/", help="root path for large-scale reusable data")

    parser.add_argument("-bs", "--batch_size", dest="batch_size",
                        help="batch size", type=int, default=100)
    parser.add_argument("-hs", "--his_size", dest="his_size",
                        help="history size", type=int, default=50)
    parser.add_argument("-is", "--impr_size", dest="impr_size",
                        help="impression size for evaluating", type=int, default=50)
    parser.add_argument("-tl", "--title_length", dest="title_length",
                        help="news title size", type=int, default=20)
    parser.add_argument("-as", "--abs_length", dest="abs_length",
                        help="news abstract length", type=int, default=40)
    parser.add_argument("-sl", "--signal_length", dest="signal_length",
                    help="length of the bert tokenized tokens", type=int, default=100)

    parser.add_argument("-hd", "--hidden_dim", dest="hidden_dim",
                    help="number of hidden states", type=int, default=384)
    parser.add_argument("-dp", "--dropout_p", dest="dropout_p",
                    help="dropout probability", type=float, default=0.2)

    parser.add_argument("-st","--step", dest="step",
                        help="save/evaluate the model every step", type=int, default=10000)
    parser.add_argument("-ck","--checkpoint", dest="checkpoint",
                        help="load the model from checkpoint before training/evaluating", type=int, default=0)

    parser.add_argument("-lr", dest="lr",
                        help="learning rate of non-bert modules", type=float, default=1e-4)
    parser.add_argument("-blr", "--bert_lr", dest="bert_lr",
                        help="learning rate of bert based modules", type=float, default=3e-5)
    parser.add_argument("-sm", "--smoothing", dest="smoothing", help="smoothing factor of tqdm", type=float, default=0.3)

    parser.add_argument("--ascend_history", dest="ascend_history", help="whether to order history by time in ascending", action='store_true', default=False)
    parser.add_argument("--no_dedup", dest="no_dedup", help="whether to disable deduplication forarticles", action='store_true', default=False)

    parser.add_argument("--num_workers", dest="num_workers", help="worker number of a dataloader", type=int, default=0)
    parser.add_argument("--shuffle", dest="shuffle", help="whether to shuffle the indices", action='store_true', default=False)
    parser.add_argument("--pin_memory", dest="pin_memory", help="whether to pin memory to speed up tensor transfer", default=True)
    parser.add_argument("--scheduler", dest="scheduler", help="choose schedule scheme for optimizer", choices=['linear'], default="linear")
    parser.add_argument("--warmup", dest="warmup", help="warmup steps of scheduler", type=int, default=10000)
    parser.add_argument("--interval", dest="interval", help="the step interval to update processing bar", default=10, type=int)

    parser.add_argument("--npratio", dest="npratio", help="the number of unclicked news to sample when training", type=int, default=4)
    parser.add_argument("--metrics", dest="metrics", help="metrics for evaluating the model", type=str, default='')

    parser.add_argument("-emb", "--embedding", dest="embedding", help="choose embedding", choices=['bert','random'], default='bert')
    parser.add_argument("-encn", "--encoderN", dest="encoderN", help="choose news encoder", choices=['cnn','rnn','npa','fim','mha','bert'], default="cnn")
    parser.add_argument("-encu", "--encoderU", dest="encoderU", help="choose user encoder", choices=['avg','attn','cnn','rnn','lstur','mha'], default="rnn")
    parser.add_argument("-slc", "--selector", dest="selector", help="choose history selector", choices=['recent','sfi'], default="sfi")
    parser.add_argument("-red", "--reducer", dest="reducer", help="choose document reducer", choices=['bm25','matching'], default="matching")
    parser.add_argument("-fus", "--fuser", dest="fuser", help="choose term fuser", choices=['union'], default="union")
    parser.add_argument("-rk", "--ranker", dest="ranker", help="choose ranker", choices=['onepass','original','cnn','knrm'], default="onepass")

    parser.add_argument("-k", dest="k", help="the number of the terms to extract from each news article", type=int, default=5)
    parser.add_argument("-thr", "--threshold", dest="threshold", help="threshold to mask terms", default=-float("inf"), type=float)

    parser.add_argument("--spadam", dest="spadam", action='store_true', default=False)
    parser.add_argument("--tb", dest="tb", action='store_true', default=False)
    parser.add_argument("--seeds", dest="seeds", default=None, type=int)

    parser.add_argument("--bert", dest="bert", help="choose bert model", choices=["bert-base-uncased"], default="bert-base-uncased")

    parser.add_argument("-hn", "--head_num", dest="head_num", help="number of multi-heads", type=int, default=12)

    parser.add_argument("-ws", "--world_size", dest="world_size", help="total number of gpus", default=0, type=int)


    args = parser.parse_args()

    args.cdd_size = args.npratio + 1
    args.metrics = "auc,mean_mrr,ndcg@5,ndcg@10".split(',') + [i for i in args.metrics.split(',') if i]
    if args.device == -1:
        args.device = 'cpu'
        args.pin_memory = False

    manager = Manager(args)
    return manager

def manual_seed(seed):
    # os.environ['PYTHONHASHSEED'] = str(seed)
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
    logger.info("Hyper Parameters are \n{}".format(info(config)))

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

    #     vocab = getVocab('data/dictionaries/vocab.pkl')
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

        return (loader_train, loader_dev)

    elif config.mode == "dev":
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

        return (loader_dev,)

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

        return (loader_test,)



def analyse(config):
    """
        analyse over MIND
    """
    mind_path = config.path + 'MIND'

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
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=manager.world_size)

        os.environ['TOKENIZERS_PARALLELISM'] = 'True'
        # manager.rank will be invoked in creating DistributedSampler
        manager.rank = rank
        # manager.device will be invoked in the model
        manager.device = rank

    else:
        # one-gpu
        manager.rank = -1

    torch.cuda.set_device(rank)


def cleanup():
    """
    shut down the distributed training process
    """
    dist.destroy_process_group()


class Partition_Sampler():
    def __init__(self, dataset, num_replicas, rank) -> None:
        super().__init__()
        len_per_worker, extra_len = divmod(len(dataset), num_replicas)
        self.dataset = dataset
        self.start = len_per_worker * rank
        self.end = self.start + len_per_worker + extra_len * (rank + 1== num_replicas)

    def __iter__(self):
        start = self.start
        end = self.end - 1

        # strip off the impression without 1 label, such impression only occurs at the start or the end
        # while 1:
        #     if not (self.dataset[end]['label'] == 1).any():
        #         end -= 1

        # while 1:
        #     if not (self.dataset[start]['label'] == 1).any():
        #         start += 1

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
        word_count = 0
        doc_count = len(documents)

        tfs = []
        df = defaultdict(int)
        for document in documents:
            tf = defaultdict(int)
            # ignore [CLS]
            for term in document[1:]:
                # ignore [PAD]
                if term != 0:
                    tf[term] += 1
                    df[term] += 1
                    word_count + 1

            tfs.append(tf)

        self.tfs = tfs

        idf = defaultdict(float)
        for term,freq in df.items():
            idf[term] = math.log((doc_count - freq + 0.5 ) / (freq + 0.5) + 1)

        self.idf = idf


    def __call__(self, documents):
        """
        compute bm25 score of each term (index) in each document and sort the terms by it
        with b=0, totally ignoring the effect of document length

        Args:
            documents: list of lists
        """
        logger.info("computing BM25 scores...")
        self._build_tf_idf(documents)

        document_length = len(documents[0])
        bm25_scores = []
        for tf in self.tfs:
            score = defaultdict(float)
            for term, freq in tf.items():
                score[term] = (self.idf[term] * freq * (self.k + 1)) / (freq + self.k)

            bm25_scores.append(dict(sorted(score.items(), key=lambda item: item[1], reverse=True)))

        sorted_documents = []
        sorted_attn_mask = []
        for i, bm25 in enumerate(bm25_scores):
            bm25_length = len(bm25) + 1
            pad_length = document_length - bm25_length
            if not i:
                sorted_documents.append([0] * document_length)
                sorted_attn_mask.append([0] * document_length)
            else:
                sorted_documents.append([101] + list(bm25.keys()) + [0]*pad_length)
                sorted_attn_mask.append([1] * bm25_length + [0] * pad_length)

        return np.asarray(sorted_documents), np.asarray(sorted_attn_mask)


class DeDuplicate(object):
    """
    mask duplicated terms in one document by attention masks
    """
    def __init__(self, k, signal_length) -> None:
        super().__init__()
        self.k = k
        self.signal_length = signal_length

    def __call__(self, documents, attn_masks):
        """
        return modified attention masks, where duplicated terms are masked
        """
        logger.info("deduplicating...")
        for i, document in enumerate(documents):
            terms = {}
            duplicated = []
            # ignore [CLS]
            for j, term in enumerate(document[1:self.signal_length]):
                if term in terms:
                    # if the term duplicates
                    duplicated.append(j)
                else:
                    terms[term] = 1

            # ignore [CLS]
            term_num = attn_masks[i][1:self.signal_length].sum()
            if term_num - len(duplicated) > self.k:
                # in case the non-duplicate terms are more than k, then mask duplicated terms
                attn_masks[i, duplicated] = 0
            else:
                # otherwise, only maks part of them
                maskable_term_num = term_num - self.k
                attn_masks[i, duplicated[:maskable_term_num]] = 0

        return documents, attn_masks


class DoNothing(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, documents, attn_masks):
        return documents, attn_masks