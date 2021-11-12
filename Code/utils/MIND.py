import re
import os
import json
import pickle
import logging

import torch
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.utils import newsample, getId2idx, tokenize, getVocab

logger = logging.getLogger(__name__)


class MINDBaseDataset(Dataset):
    def __init__(self, manager, file_directory, news=True, behaviors=True) -> None:
        super().__init__()
        self.his_size = manager.his_size
        self.impr_size = manager.impr_size

        self.k = manager.k
        self.signal_length = manager.signal_length

        self.npratio = manager.npratio
        self.shuffle_pos = manager.shuffle_pos
        self.descend_history = manager.descend_history

        self.reducer = manager.reducer
        self.granularity = manager.granularity

        pat = re.search('/(MIND(\w*?)_(.*))/', file_directory)
        self.file_name = pat.group(1)
        self.scale = pat.group(2)
        self.mode = pat.group(3)

        self.bert = manager.bert

        cache_directory = "data/cache/MIND"

        if behaviors:
            if self.mode == "train":
                behav_cache_directory = "/".join([cache_directory, "behaviors", self.file_name])
            else:
                behav_cache_directory = "/".join([cache_directory, "behaviors", self.file_name, str(self.impr_size)])

            if manager.mode == "recall":
                self.behav_cache_path = "/".join([behav_cache_directory, "recall.pkl"])
            else:
                self.behav_cache_path = "/".join([behav_cache_directory, "behaviors.pkl"])
            # initialize all caches on master node
            if manager.rank in [-1, 0]:
                # only do this in the basic modes
                if not os.path.exists(self.behav_cache_path):
                    self.behaviors_file = file_directory + "behaviors.tsv"
                    logger.info("encoding user behaviors of {}...".format(self.behaviors_file))
                    os.makedirs(behav_cache_directory, exist_ok=True)
                    try:
                        self.nid2index = getId2idx("data/dictionaries/nid2idx_{}_{}.json".format(self.scale, self.mode))
                    except FileNotFoundError:
                        manager.construct_nid2idx(mode=self.mode)
                        self.nid2index = getId2idx("data/dictionaries/nid2idx_{}_{}.json".format(self.scale, self.mode))
                    try:
                        self.uid2index = getId2idx("data/dictionaries/uid2idx_{}.json".format(self.scale))
                    except FileNotFoundError:
                        manager.construct_uid2idx()
                        self.uid2index = getId2idx("data/dictionaries/uid2idx_{}.json".format(self.scale))
                    self.init_behaviors()

            # synchronize all processes
            if manager.world_size > 1:
                dist.barrier()

            # logger.info("process NO.{} loading cached user behavior from {}".format(manager.rank, self.behav_cache_path))
            with open(self.behav_cache_path, "rb") as f:
                behaviors = pickle.load(f)
                # target at one news
                if manager.news is not None:
                    assert manager.mode == 'inspect', "target news only available in INSPECT mode"
                    try:
                        self.nid2index = getId2idx("data/dictionaries/nid2idx_{}_{}.json".format(self.scale, self.mode))
                    except FileNotFoundError:
                        manager.construct_nid2idx(mode=self.mode)
                        self.nid2index = getId2idx("data/dictionaries/nid2idx_{}_{}.json".format(self.scale, self.mode))

                    logger.info("extracting users who browsed news {}".format(manager.news))

                    news = self.nid2index[manager.news]

                    imprs = []

                    self.histories = behaviors['histories']
                    self.uindexes = behaviors['uindexes']

                    for i in behaviors['imprs']:
                        impr_index = i[0]
                        # limit the length of user history to his_size
                        if news in self.histories[impr_index][:self.his_size]:
                            imprs.append(i)
                    self.imprs = imprs

                else:
                    for k,v in behaviors.items():
                        setattr(self, k, v)

        if news:
            news_cache_directory = "/".join([cache_directory, "news", manager.get_bert_for_cache(), self.file_name])
            # used in manager.inspect()
            self.news_cache_directory = news_cache_directory + '/'
            self.news_cache_path = "/".join([news_cache_directory, manager.get_news_file_for_load()])
            self.pad_token_id = manager.get_special_token_id('[PAD]')
            self.sep_token_id = manager.get_special_token_id('[SEP]')

            if manager.rank in [-1, 0]:
                # FIXME: gather all reducer functions
                if not os.path.exists(self.news_cache_path):
                    self.news_file = file_directory + "news.tsv"
                    logger.info("encoding news of {}...".format(self.news_file))
                    os.makedirs(news_cache_directory, exist_ok=True)

                    from transformers import AutoTokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(manager.get_bert_for_load(), cache_dir=manager.path + "bert_cache/")
                    self.bert = manager.bert
                    self.max_token_length = 512
                    self.max_reduction_length = 30
                    self.init_news()

            # synchronize all processes
            if manager.world_size > 1:
                dist.barrier()
                
            # logger.info("process NO.{} loading cached news tokenization from {}".format(manager.rank, self.news_cache_path))
            with open(self.news_cache_path, "rb") as f:
                news = pickle.load(f)
                self.encoded_news = news["encoded_news"][:, :self.signal_length]
                self.attn_mask = news["attn_mask"][:, :self.signal_length]
                if self.granularity in ["avg","sum"]:
                    self.subwords = news["subwords_all"][:, :self.signal_length]
                elif self.granularity == "first":
                    self.subwords = news["subwords_first"][:, :self.signal_length]
                else:
                    self.subwords = None

            if self.reducer in ["bm25", "entity", "first", "keyword"]:
                # prune encoded news index
                self.encoded_news = self.encoded_news[:, :self.k + 1]
                self.attn_mask = self.attn_mask[:, :self.k + 1]
                # [CLS]
                if self.subwords is not None:
                    self.subwords = self.subwords[:, :self.k + 1]

                with open(self.news_cache_directory + "news.pkl", "rb") as f:
                    news = pickle.load(f)
                    self.encoded_news_original = news["encoded_news"][:, :self.signal_length]
                    self.attn_mask_original = news["attn_mask"][:, :self.signal_length]
                    if self.granularity in ["avg","sum"]:
                        self.subwords_original = news["subwords_all"][:, :self.signal_length]
                    elif self.granularity == "first":
                        self.subwords_original = news["subwords_first"][:, :self.signal_length]
                    else:
                        self.subwords_original = None

            # refine
            if manager.reducer == "matching":
                if not manager.no_dedup:
                    from utils.utils import DeDuplicate
                    refiner = DeDuplicate(manager)
                else:
                    refiner = None
            elif manager.reducer in ["bm25", "none", "entity", "first", "keyword"]:
                refiner = None
            elif manager.reducer == "bow":
                from utils.utils import CountFreq
                refiner = CountFreq(manager)

            # set the last token of a sequence to [SEP]
            sep_pos = self.encoded_news[:, -1] != self.pad_token_id
            self.encoded_news[:, -1] = self.sep_token_id * sep_pos
            if hasattr(self, "encoded_news_original"):
                sep_pos = self.encoded_news_original[:, -1] != 0
                self.encoded_news_original[:, -1] = self.sep_token_id * sep_pos

            self.init_refinement(refiner)


    def init_news(self):
        """
            parse news text into tokens, and save in the file, no assignment to self

            1. encode news text to tokens
            2. rerank words in the news text by bm25
            3. get subword indices
            4. get entities
        """
        # tokenize once, remove punctuations in BM25
        articles = [""]
        entities = [""]
        keywords = [""]
        with open(self.news_file, "r", encoding="utf-8") as rd:
            for idx in tqdm(rd, ncols=120, leave=True):
                nid, vert, subvert, title, ab, url, title_entity, abs_entity = idx.strip("\n").split("\t")
                article = " ".join([title, ab, subvert])
                articles.append(article)

                entity_dic = dict()
                title_entity = json.loads(title_entity)
                abs_entity = json.loads(abs_entity)
                for entity in title_entity + abs_entity:
                    surface_forms = entity["SurfaceForms"]
                    if len(surface_forms):
                        entity_dic[surface_forms[0].lower()] = 1

                if len(entity_dic) == 0:
                    words = re.sub("[.&*()+=/\<>,!?;:~`@#$%^]", '', article).split()
                    entities.append(' '.join(words[:self.max_reduction_length]))
                else:
                    entities.append(' '.join(list(entity_dic.keys())))

        # load pre-defined keywords
        try:
            with open(self.file_directory + "keywords.tsv", "r", encoding="utf-8") as rd:
                for idx in tqdm(rd, ncols=120, leave=True):
                    keyword = idx.strip("\n")
                    keywords.append(keyword)
            no_keyword = False
        except:
            no_keyword = True
            logger.warning("no pre-defined keywords found")

        # initialize other kind of reducer here
        # rank words according to reduction rules
        from utils.utils import BM25
        bm25 = BM25()
        articles_bm25 = bm25(articles)

        def parse_texts_bert(tokenizer, texts, news_path, max_length):
            """
            convert texts to tokens indices and get subword indices
            """
            text_toks = []
            attention_masks = []
            subwords_all = []
            subwords_first = []
            for i, text in enumerate(tqdm(texts, ncols=120, leave=True)):
                if i == 0:
                    token_ids = [self.pad_token_id] * max_length
                    attn_mask = [0] * max_length
                    subword_first = [[0,0]] * max_length
                    subword_all = [[0,0]] * max_length

                else:
                    token_ouput = tokenizer(text, padding='max_length', truncation=True, max_length=max_length)
                    token_ids = token_ouput['input_ids']
                    attn_mask = token_ouput['attention_mask']
                    tokens = tokenizer.convert_ids_to_tokens(token_ids)

                    # maintain subword entry
                    subword_all = []
                    # mask subword entry
                    subword_first = []

                    i = -1
                    j = -1
                    for token in tokens:
                        if token == '[PAD]':
                            subword_all.append([0,0])
                            subword_first.append([0,0])

                        elif token.startswith("##"):
                            j += 1
                            subword_all.append([i,j])
                            subword_first.append([0,0])

                        else:
                            i += 1
                            j += 1
                            subword_all.append([i,j])
                            subword_first.append([i,j])

                text_toks.append(token_ids)
                attention_masks.append(attn_mask)
                subwords_all.append(subword_all)
                subwords_first.append(subword_first)

            # encode news
            encoded_news = np.asarray(text_toks)
            attn_mask = np.asarray(attention_masks)

            subwords_all = np.asarray(subwords_all)
            subwords_first = np.asarray(subwords_first)

            with open(news_path, "wb") as f:
                pickle.dump(
                    {
                        "encoded_news": encoded_news,
                        "subwords_first": subwords_first,
                        "subwords_all": subwords_all,
                        "attn_mask": attn_mask
                    },
                    f
                )

        def parse_texts_wordpiece(tokenizer, texts, news_path, max_length):
            """
            convert texts to tokens indices and get subword indices
            """
            text_toks = []
            attention_masks = []
            subwords_all = []
            subwords_first = []
            for i, text in enumerate(texts):
                if i == 0:
                    token_ids = [self.pad_token_id] * max_length
                    attn_mask = [0] * max_length
                    subword_first = [[0,0]] * max_length
                    subword_all = [[0,0]] * max_length

                else:
                    token_ouput = tokenizer(text, padding='max_length', truncation=True, max_length=max_length)
                    token_ids = token_ouput['input_ids']
                    attn_mask = token_ouput['attention_mask']

                    tokens = tokenizer.convert_ids_to_tokens(token_ids)

                    # maintain subword entry
                    subword_all = []
                    # mask subword entry
                    subword_first = []

                    i = -1
                    j = -1
                    for index,token in enumerate(tokens):
                        if token == '[PAD]':
                            subword_all.append([0,0])
                            subword_first.append([0,0])

                        # not subword
                        elif index in [0,1] or token.startswith("Ġ") or token in r"[.&*()+=/\<>,!?;:~`@#$%^]":
                            i += 1
                            j += 1
                            subword_all.append([i,j])
                            subword_first.append([i,j])

                        # subword
                        else:
                            j += 1
                            subword_all.append([i,j])
                            subword_first.append([0,0])

                text_toks.append(token_ids)
                attention_masks.append(attn_mask)
                subwords_all.append(subword_all)
                subwords_first.append(subword_first)

            # encode news
            encoded_news = np.asarray(text_toks)
            attn_mask = np.asarray(attention_masks)

            subwords_all = np.asarray(subwords_all)
            subwords_first = np.asarray(subwords_first)

            with open(news_path, "wb") as f:
                pickle.dump(
                    {
                        "encoded_news": encoded_news,
                        "subwords_first": subwords_first,
                        "subwords_all": subwords_all,
                        "attn_mask": attn_mask
                    },
                    f
                )

        def parse_texts_sentencepiece(tokenizer, texts, news_path, max_length):
            """
            convert texts to tokens indices and get subword indices
            """
            text_toks = []
            attention_masks = []
            subwords_all = []
            subwords_first = []
            for i, text in enumerate(texts):
                if i == 0:
                    token_ids = [self.pad_token_id] * max_length
                    attn_mask = [0] * max_length
                    subword_first = [[0,0]] * max_length
                    subword_all = [[0,0]] * max_length

                else:
                    token_ouput = tokenizer(text, padding='max_length', truncation=True, max_length=max_length)
                    token_ids = token_ouput['input_ids']
                    attn_mask = token_ouput['attention_mask']

                    tokens = tokenizer.convert_ids_to_tokens(token_ids)

                    # maintain subword entry
                    subword_all = []
                    # mask subword entry
                    subword_first = []

                    i = -1
                    j = -1
                    for index,token in enumerate(tokens):
                        if token == '[PAD]':
                            subword_all.append([0,0])
                            subword_first.append([0,0])

                        # not subword
                        # index==0: [CLS], index==1: the first word
                        elif index in [0,1] or token.startswith("▁") or token in r"[.&*()+=/\<>,!?;:~`@#$%^]":
                            i += 1
                            j += 1
                            subword_all.append([i,j])
                            subword_first.append([i,j])

                        # subword
                        else:
                            j += 1
                            subword_all.append([i,j])
                            subword_first.append([0,0])

                text_toks.append(token_ids)
                attention_masks.append(attn_mask)
                subwords_all.append(subword_all)
                subwords_first.append(subword_first)

            # encode news
            encoded_news = np.asarray(text_toks)
            attn_mask = np.asarray(attention_masks)

            subwords_all = np.asarray(subwords_all)
            subwords_first = np.asarray(subwords_first)

            with open(news_path, "wb") as f:
                pickle.dump(
                    {
                        "encoded_news": encoded_news,
                        "subwords_first": subwords_first,
                        "subwords_all": subwords_all,
                        "attn_mask": attn_mask
                    },
                    f
                )

        def parse_texts_xformer(tokenizer, texts, news_path, max_length):
            """
            convert texts to tokens indices and get subword indices
            """
            # manually set padding token
            # tokenizer.pad_token = tokenizer.eos_token

            text_toks = []
            attention_masks = []
            # subwords_all = []
            # subwords_first = []
            for i, text in enumerate(tqdm(texts, ncols=120, leave=True)):
                token_ouput = tokenizer(text, padding='max_length', truncation=True, max_length=max_length - 1)
                token_ouput["input_ids"].insert(0, 2)
                token_ouput["attention_mask"].insert(0, 1)

                token_ids = token_ouput['input_ids']
                attn_mask = token_ouput['attention_mask']
                # tokens = tokenizer.convert_ids_to_tokens(token_ids)

                # maintain subword entry
                # subword_all = []
                # mask subword entry
                # subword_first = []
                # i = -1
                # j = -1
                # for token in tokens:
                #     if token == '[PAD]':
                #         subword_all.append([0,0])
                #         subword_first.append([0,0])

                #     elif token.startswith("##"):
                #         j += 1
                #         subword_all.append([i,j])
                #         subword_first.append([0,0])

                #     else:
                #         i += 1
                #         j += 1
                #         subword_all.append([i,j])
                #         subword_first.append([i,j])

                text_toks.append(token_ids)
                attention_masks.append(attn_mask)
                # subwords_all.append(subword_all)
                # subwords_first.append(subword_first)

            # encode news
            encoded_news = np.asarray(text_toks)
            attn_mask = np.asarray(attention_masks)

            # subwords_all = np.asarray(subwords_all)
            # subwords_first = np.asarray(subwords_first)

            with open(news_path, "wb") as f:
                pickle.dump(
                    {
                        "encoded_news": encoded_news,
                        "attn_mask": attn_mask,
                        # "subwords_first": subwords_first,
                        # "subwords_all": subwords_all,
                    },
                    f
                )

        if self.bert in ['bert', 'unilm']:
            logger.info("tokenizing news...")
            parse_texts_bert(self.tokenizer, articles, self.news_cache_directory + "news.pkl", self.max_token_length)
            logger.info("tokenizing bm25 ordered news...")
            parse_texts_bert(self.tokenizer, articles_bm25, self.news_cache_directory + "bm25.pkl", self.max_reduction_length)
            logger.info("tokenizing entities...")
            parse_texts_bert(self.tokenizer, entities, self.news_cache_directory + "entity.pkl", self.max_reduction_length)
            if not no_keyword:
                logger.info("tokenizing keywords...")
                parse_texts_bert(self.tokenizer, keywords, self.news_cache_directory + "keyword.pkl", self.max_reduction_length)

        elif self.bert in ['deberta']:
            logger.info("tokenizing news...")
            parse_texts_wordpiece(self.tokenizer, articles, self.news_cache_directory + "news.pkl", self.max_token_length)
            logger.info("tokenizing bm25 ordered news...")
            parse_texts_wordpiece(self.tokenizer, articles_bm25, self.news_cache_directory + "bm25.pkl", self.max_reduction_length)
            logger.info("tokenizing entities...")
            parse_texts_wordpiece(self.tokenizer, entities, self.news_cache_directory + "entity.pkl", self.max_reduction_length)
            if not no_keyword:
                logger.info("tokenizing keywords...")
                parse_texts_wordpiece(self.tokenizer, keywords, self.news_cache_directory + "keyword.pkl", self.max_reduction_length)

        # elif self.bert in ['']:
        #     logger.info("tokenizing news...")
        #     parse_texts_sentencepiece(self.tokenizer, articles, self.news_cache_directory + "news.pkl", self.max_token_length)
        #     logger.info("tokenizing bm25 ordered news...")
        #     parse_texts_sentencepiece(self.tokenizer, articles_bm25, self.news_cache_directory + "bm25.pkl", self.max_reduction_length)
        #     logger.info("tokenizing entities...")
        #     parse_texts_sentencepiece(self.tokenizer, entities, self.news_cache_directory + "entity.pkl", self.max_reduction_length)
        #     if not no_keyword:
        #         logger.info("tokenizing keywords...")
        #         parse_texts_sentencepiece(self.tokenizer, keywords, self.news_cache_directory + "keyword.pkl", self.max_reduction_length)

        # avoid processing subwords
        # elif self.bert in ['reformer', "tiny", "longformer", "synthesizer", "funnel", "bigbird"]:
        else:
            logger.info("tokenizing news...")
            parse_texts_xformer(self.tokenizer, articles, self.news_cache_directory + "news.pkl", self.max_token_length)
            logger.info("tokenizing bm25 ordered news...")
            parse_texts_xformer(self.tokenizer, articles_bm25, self.news_cache_directory + "bm25.pkl", self.max_reduction_length)
            logger.info("tokenizing entities...")
            parse_texts_xformer(self.tokenizer, entities, self.news_cache_directory + "entity.pkl", self.max_reduction_length)


    def init_behaviors(self):
        """
            init behavior logs given behaviors file.
        """
        # list of list of history news index
        histories = []
        # list of user index
        uindexes = []
        # list of impression indexes
        # self.impr_indexes = []

        impr_index = 0

        # only store positive behavior
        if self.mode == "train":
            # list of lists, each list represents a
            imprs = []
            negatives = []
            with open(self.behaviors_file, "r", encoding="utf-8") as rd:
                for idx in tqdm(rd, ncols=120, leave=True):
                    _, uid, time, history, impr = idx.strip("\n").split("\t")

                    history = [self.nid2index[i] for i in history.split()]

                    impr_news = [self.nid2index[i.split("-")[0]] for i in impr.split()]
                    labels = [int(i.split("-")[1]) for i in impr.split()]

                    # user will always in uid2index
                    uindex = self.uid2index[uid]
                    # store negative samples of each impression
                    negative = []

                    for news, label in zip(impr_news, labels):
                        if label == 1:
                            imprs.append((impr_index, news))
                        else:
                            negative.append(news)

                    # 1 impression correspond to 1 of each of the following properties
                    histories.append(history)
                    negatives.append(negative)
                    uindexes.append(uindex)

                    impr_index += 1

            save_dict = {
                "imprs": imprs,
                "histories": histories,
                "negatives": negatives,
                "uindexes": uindexes
            }

            with open(self.behav_cache_path, "wb") as f:
                pickle.dump(save_dict, f)

        # store every behavior
        elif self.mode == "dev":
            # list of every cdd news index along with its impression index and label
            imprs = []
            with open(self.behaviors_file, "r", encoding="utf-8") as rd:
                for idx in tqdm(rd, ncols=120, leave=True):
                    _, uid, time, history, impr = idx.strip("\n").split("\t")

                    history = [self.nid2index[i] for i in history.split()]

                    impr_news = [self.nid2index[i.split("-")[0]] for i in impr.split()]
                    labels = [int(i.split("-")[1]) for i in impr.split()]
                    # user will always in uid2index
                    uindex = self.uid2index[uid]

                    # store every impression
                    for i in range(0, len(impr_news), self.impr_size):
                        imprs.append((impr_index, impr_news[i:i+self.impr_size], labels[i:i+self.impr_size]))

                    # 1 impression correspond to 1 of each of the following properties
                    histories.append(history)
                    uindexes.append(uindex)

                    impr_index += 1

            save_dict = {
                "imprs": imprs,
                "histories": histories,
                "uindexes": uindexes
            }

            with open(self.behav_cache_path, "wb") as f:
                pickle.dump(save_dict, f)

        # store every behavior
        elif self.mode == "test":
            # list of every cdd news index along with its impression index and label
            imprs = []
            with open(self.behaviors_file, "r", encoding="utf-8") as rd:
                for idx in tqdm(rd, ncols=120, leave=True):
                    _, uid, time, history, impr = idx.strip("\n").split("\t")

                    history = [self.nid2index[i] for i in history.split()]

                    impr_news = [self.nid2index[i] for i in impr.split()]
                    # user will always in uid2index
                    uindex = self.uid2index[uid]

                    # store every impression
                    for i in range(0, len(impr_news), self.impr_size):
                        imprs.append((impr_index, impr_news[i:i+self.impr_size]))

                    # 1 impression correspond to 1 of each of the following properties
                    histories.append(history)
                    uindexes.append(uindex)

                    impr_index += 1

            save_dict = {
                "imprs": imprs,
                "histories": histories,
                "uindexes": uindexes
            }

            with open(self.behav_cache_path, "wb") as f:
                pickle.dump(save_dict, f)


    def init_refinement(self, refiner):
        """
            token level refinement

            1. dedplicate for matching reducer
            2. count frequency for bog reducer
        """
        if refiner is None:
            return

        if self.reducer == "matching":
            refined_news, refined_mask = refiner(self.encoded_news, self.attn_mask)
            self.attn_mask_dedup = refined_mask

        elif self.reducer == "bow":
            refined_news, refined_mask = refiner(self.encoded_news, self.attn_mask)
            self.encoded_news = refined_news
            self.attn_mask = refined_mask



class MIND(MINDBaseDataset):
    def __init__(self, manager, file_directory):
        """ Map Style Dataset for MIND

        Args:
            manager(dict): pre-defined dictionary of hyper parameters
            file_directory(str): directory to news and behaviors file
        """
        super().__init__(manager, file_directory)

        if manager.bert == "reformer":
            with open('/'.join(["data/cache/MIND/news/bert", self.file_name, "news.pkl"]), "rb") as f:
                news = pickle.load(f)
                self.encoded_news_original = news["encoded_news"][:, :self.signal_length]
                self.attn_mask_original = news["attn_mask"][:, :self.signal_length]
                sep_pos = self.encoded_news_original[:, -1] != 0
                self.encoded_news_original[:, -1] = 102 * sep_pos


    def __len__(self):
        """
            return length of the whole dataset
        """
        return len(self.imprs)


    def __getitem__(self,index):
        """ return data
        Args:
            index: the index for stored impression

        Returns:
            back_dic: dictionary of data slice
        """

        impr = self.imprs[index] # (impression_index, news_index)
        impr_index = impr[0]
        impr_news = impr[1]
        user_index = self.uindexes[impr_index]

        # each time called to return positive one sample and its negative samples
        if self.mode == "train":
            # user"s unclicked news in the same impression
            negs = self.negatives[impr_index]
            neg_list, neg_num = newsample(negs, self.npratio)

            cdd_ids = [impr_news] + neg_list
            cdd_size = self.npratio + 1

            label = np.asarray([1] + [0] * self.npratio)

            if self.shuffle_pos:
                s = np.arange(0, len(label), 1)
                np.random.shuffle(s)
                cdd_ids = np.asarray(cdd_ids)[s]
                label = np.asarray(label)[s]

            label = np.arange(0, len(cdd_ids), 1)[label == 1][0]

            # pad in his_id, not in histories
            his_ids = self.histories[impr_index][:self.his_size]
            # true means the corresponding history news is padded
            his_mask = np.zeros((self.his_size, 1))
            his_length = len(his_ids)
            if his_length == 0:
                his_mask[0] = 1
            else:
                his_mask[:his_length] = 1

            cdd_mask = np.zeros((cdd_size, 1))
            cdd_mask[:neg_num + 1] = 1

            if self.descend_history:
                his_ids = his_ids[::-1] + [0] * (self.his_size - len(his_ids))
            else:
                his_ids = his_ids + [0] * (self.his_size - len(his_ids))

            cdd_encoded_index = self.encoded_news[cdd_ids]
            cdd_attn_mask = self.attn_mask[cdd_ids]
            his_encoded_index = self.encoded_news[his_ids]
            his_attn_mask = self.attn_mask[his_ids]

            back_dic = {
                "user_id": user_index,
                "cdd_id": np.asarray(cdd_ids),
                "his_id": np.asarray(his_ids),
                "cdd_encoded_index": cdd_encoded_index,
                "his_encoded_index": his_encoded_index,
                "cdd_attn_mask": cdd_attn_mask,
                "his_attn_mask": his_attn_mask,
                "cdd_mask": cdd_mask,
                "his_mask": his_mask,
                "label": label
            }

            # word-level
            if self.subwords is not None:
                if self.reducer in ["bm25","entity","first"]:
                    # subwords of history news don't accord with candidate news
                    cdd_subword_index = self.subwords_original[cdd_ids]
                    his_subword_index = self.subwords[his_ids]
                else:
                    # matching
                    cdd_subword_index = self.subwords[cdd_ids]
                    his_subword_index = self.subwords[his_ids]
                back_dic["cdd_subword_index"] = cdd_subword_index
                back_dic["his_subword_index"] = his_subword_index

            if self.reducer == "matching" and hasattr(self, "attn_mask_dedup"):
                his_attn_mask_dedup = self.attn_mask_dedup[his_ids]
                back_dic["his_refined_mask"] = his_attn_mask_dedup

            # when running reformer, the candidate is processed by a normal bert
            elif self.reducer in ["bm25","entity","first"] or self.bert == "reformer":
                back_dic["cdd_encoded_index"] = self.encoded_news_original[cdd_ids]
                back_dic["cdd_attn_mask"] = self.attn_mask_original[cdd_ids]

            elif self.reducer == "bow":
                back_dic["his_refined_mask"] = back_dic["his_attn_mask"]

            return back_dic

        # each time called return one sample, and no labels
        elif self.mode == "dev":
            cdd_ids = impr_news
            cdd_size = len(cdd_ids)

            # pad in his_id, not in histories
            his_ids = self.histories[impr_index][:self.his_size]
            # true means the corresponding history news is padded
            his_mask = np.zeros((self.his_size, 1))
            his_length = len(his_ids)
            if his_length == 0:
                his_mask[0] = 1
            else:
                his_mask[:his_length] = 1

            if self.descend_history:
                his_ids = his_ids[::-1] + [0] * (self.his_size - len(his_ids))
            else:
                his_ids = his_ids + [0] * (self.his_size - len(his_ids))

            label = impr[2]

            cdd_encoded_index = self.encoded_news[cdd_ids]
            cdd_attn_mask = self.attn_mask[cdd_ids]
            his_encoded_index = self.encoded_news[his_ids]
            his_attn_mask = self.attn_mask[his_ids]

            back_dic = {
                "impr_index": impr_index + 1,
                "user_id": user_index,
                "cdd_id": np.asarray(cdd_ids),
                "his_id": np.asarray(his_ids),
                "cdd_encoded_index": cdd_encoded_index,
                "his_encoded_index": his_encoded_index,
                "cdd_attn_mask": cdd_attn_mask,
                "his_attn_mask": his_attn_mask,
                "his_mask": his_mask,
                "label": np.asarray(label)
            }

            if self.subwords is not None:
                if self.reducer in ["bm25","entity","first"]:
                    cdd_subword_index = self.subwords_original[cdd_ids]
                    his_subword_index = self.subwords[his_ids]
                else:
                    cdd_subword_index = self.subwords[cdd_ids]
                    his_subword_index = self.subwords[his_ids]
                back_dic["cdd_subword_index"] = cdd_subword_index
                back_dic["his_subword_index"] = his_subword_index

            if self.reducer == "matching":
                his_attn_mask_dedup = self.attn_mask_dedup[his_ids]
                back_dic["his_refined_mask"] = his_attn_mask_dedup

            elif self.reducer in ["bm25","entity","first"] or self.bert == "reformer":
                back_dic["cdd_encoded_index"] = self.encoded_news_original[cdd_ids]
                back_dic["cdd_attn_mask"] = self.attn_mask_original[cdd_ids]

            elif self.reducer == "bow":
                back_dic["his_refined_mask"] = back_dic["his_attn_mask"]

            return back_dic

        elif self.mode == "test":
            cdd_ids = impr_news
            cdd_size = len(cdd_ids)

            # pad in his_id, not in histories
            his_ids = self.histories[impr_index][:self.his_size]
            # true means the corresponding history news is padded
            his_mask = np.zeros((self.his_size, 1))
            his_length = len(his_ids)
            if his_length == 0:
                his_mask[0] = 1
            else:
                his_mask[:his_length] = 1

            if self.descend_history:
                his_ids = his_ids + [0] * (self.his_size - len(his_ids))
            else:
                his_ids = his_ids[::-1] + [0] * (self.his_size - len(his_ids))


            cdd_encoded_index = self.encoded_news[cdd_ids]
            cdd_attn_mask = self.attn_mask[cdd_ids]
            his_encoded_index = self.encoded_news[his_ids]
            his_attn_mask = self.attn_mask[his_ids]

            back_dic = {
                "impr_index": impr_index + 1,
                "user_id": user_index,
                "cdd_id": np.asarray(cdd_ids),
                "his_id": np.asarray(his_ids),
                "cdd_encoded_index": cdd_encoded_index,
                "his_encoded_index": his_encoded_index,
                "cdd_attn_mask": cdd_attn_mask,
                "his_attn_mask": his_attn_mask,
                "his_mask": his_mask,
            }

            if self.subwords is not None:
                if self.reducer in ["bm25","entity","first"]:
                    cdd_subword_index = self.subwords_original[cdd_ids]
                    his_subword_index = self.subwords[his_ids]
                else:
                    cdd_subword_index = self.subwords[cdd_ids]
                    his_subword_index = self.subwords[his_ids]
                back_dic["cdd_subword_index"] = cdd_subword_index
                back_dic["his_subword_index"] = his_subword_index

            if self.reducer == "matching":
                his_attn_mask_dedup = self.attn_mask_dedup[his_ids]
                back_dic["his_refined_mask"] = his_attn_mask_dedup

            elif self.reducer in ["bm25","entity","first"] or self.bert == "reformer":
                back_dic["cdd_encoded_index"] = self.encoded_news_original[cdd_ids]
                back_dic["cdd_attn_mask"] = self.attn_mask_original[cdd_ids]

            elif self.reducer == "bow":
                back_dic["his_refined_mask"] = back_dic["his_attn_mask"]

            return back_dic

        else:
            raise ValueError("Mode {} not defined".format(self.mode))



class MIND_news(MINDBaseDataset):
    def __init__(self, manager, file_directory):
        """ Map Dataset for MIND, return each news in news.tsv

        Args:
            manager(dict): pre-defined dictionary of hyper parameters
            file_directory(str): directory to news and behaviors file
        """
        super().__init__(manager, file_directory ,behaviors=False)

        if manager.bert == "reformer":
            with open('/'.join(["data/cache/MIND/news/bert", self.file_name, "news.pkl"]), "rb") as f:
                news = pickle.load(f)
                self.encoded_news = news["encoded_news"][:, :self.signal_length]
                self.attn_mask = news["attn_mask"][:, :self.signal_length]
                sep_pos = self.encoded_news[:, -1] != 0
                self.encoded_news[:, -1] = 102 * sep_pos


    def init_refinement(self, refiner):
        """
            Override the original method to avoid adding an extra attn_mask_dedup to the dataset;
            Instead, only process bog refiner

            bow -> count
        """
        if refiner is None:
            return
        if self.reducer == "bow":
            refined_news, refined_mask = refiner(self.encoded_news, self.attn_mask)
            self.encoded_news = refined_news
            self.attn_mask = refined_mask


    def __len__(self):
        """
            return length of the whole dataset
        """
        return len(self.encoded_news)


    def __getitem__(self, idx):
        """ parse behavior log No.idx to training example

        Args:
            idx (int): impression index, start from zero

        Returns:
            dict of training data, including |npratio+1| cdd news word vector, |his_size+1| his news word vector etc.
        """

        cdd_encoded_index = self.encoded_news[[idx]]
        cdd_attn_mask = self.attn_mask[[idx]]

        back_dic = {
            "cdd_id": idx,
            "cdd_encoded_index": cdd_encoded_index,
            "cdd_attn_mask": cdd_attn_mask,
        }

        if self.subwords is not None:
            cdd_subword_index = self.subwords[[idx]]
            back_dic["cdd_subword_index"] = cdd_subword_index

        return back_dic



class MIND_history(MINDBaseDataset):
    """ Map Style Dataset for MIND, only load history behavior

    Args:
        manager(dict): pre-defined dictionary of hyper parameters
        file_directory(str): directory to news and behaviors file
    """

    def __init__(self, manager, file_directory):
        if not manager.case:
            super().__init__(manager, file_directory)

        else:
            # if case study
            super().__init__(manager, file_directory)
            self.behaviors_file = file_directory + "behaviors.tsv"
            try:
                self.nid2index = getId2idx("data/dictionaries/nid2idx_{}_{}.json".format(self.scale, self.mode))
            except FileNotFoundError:
                manager.construct_nid2idx(mode=self.mode)
                self.nid2index = getId2idx("data/dictionaries/nid2idx_{}_{}.json".format(self.scale, self.mode))
            try:
                self.uid2index = getId2idx("data/dictionaries/uid2idx_{}.json".format(self.scale))
            except FileNotFoundError:
                manager.construct_uid2idx()
                self.uid2index = getId2idx("data/dictionaries/uid2idx_{}.json".format(self.scale))

            self.refresh_behaviors()

            if manager.news is not None:
                assert manager.mode == 'inspect', "target news only available in INSPECT mode"
                logger.info("extracting users who browsed news {}".format(manager.news))

                news = self.nid2index[manager.news]
                imprs = []

                for i in self.imprs:
                    impr_index = i[0]
                    # limit the length of user history to his_size
                    if news in self.histories[impr_index][:self.his_size]:
                        imprs.append(i)
                self.imprs = imprs


    def refresh_behaviors(self):
        """
            init behavior logs given behaviors file.
        """
        # list of list of history news index
        histories = []
        # list of user index
        uindexes = []
        impr_index = 0
        # list of every cdd news index along with its impression index and label
        imprs = []

        with open(self.behaviors_file, "r", encoding="utf-8") as rd:
            for idx in tqdm(rd, ncols=120, leave=True):
                _, uid, time, history, impr = idx.strip("\n").split("\t")

                history = [self.nid2index[i] for i in history.split()]

                impr_news = [self.nid2index[i.split("-")[0]] for i in impr.split()]
                # user will always in uid2index
                uindex = self.uid2index[uid]

                # store every impression
                for i in range(0, len(impr_news), self.impr_size):
                    imprs.append((impr_index, impr_news[i:i+self.impr_size]))

                # 1 impression correspond to 1 of each of the following properties
                histories.append(history)
                uindexes.append(uindex)

                impr_index += 1

        self.imprs = imprs
        self.histories = histories
        self.uindexes = uindexes


    def __len__(self):
        """
            return length of the whole dataset
        """
        return len(self.imprs)


    def __getitem__(self,index):
        """ return data
        Args:
            index: the index for stored impression

        Returns:
            back_dic: dictionary of data slice
        """

        impr = self.imprs[index] # (impression_index, news_index)
        impr_index = impr[0]
        user_index = self.uindexes[impr_index]

        # pad in his_id, not in histories
        his_ids = self.histories[impr_index][:self.his_size]
        # true means the corresponding history news is padded
        his_mask = np.zeros((self.his_size, 1))
        his_length = len(his_ids)
        if his_length == 0:
            his_mask[0] = 1
        else:
            his_mask[:his_length] = 1

        if self.descend_history:
            his_ids = his_ids[::-1] + [0] * (self.his_size - len(his_ids))
        else:
            his_ids = his_ids + [0] * (self.his_size - len(his_ids))

        his_encoded_index = self.encoded_news[his_ids]
        his_attn_mask = self.attn_mask[his_ids]

        back_dic = {
            "impr_index": impr_index + 1,
            "user_id": user_index,
            "his_id": np.asarray(his_ids),
            "his_encoded_index": his_encoded_index,
            "his_attn_mask": his_attn_mask,
            "his_mask": his_mask,
        }

        if self.subwords is not None:
            if self.reducer in ["bm25","entity","first"]:
                his_subword_index = self.subwords[his_ids]
            else:
                his_subword_index = self.subwords[his_ids]
            back_dic["his_subword_index"] = his_subword_index

        if self.reducer == "matching":
            his_attn_mask_dedup = self.attn_mask_dedup[his_ids]
            back_dic["his_refined_mask"] = his_attn_mask_dedup

        elif self.reducer in ["bm25","entity","first"]:
            back_dic["his_attn_mask"] = back_dic["his_attn_mask"]

        elif self.reducer == "bow":
            back_dic["his_refined_mask"] = back_dic["his_attn_mask"]

        return back_dic



class MIND_recall(MINDBaseDataset):
    """ Map Style Dataset for MIND

    Args:
        manager(dict): pre-defined dictionary of hyper parameters
        file_directory(str): directory to news and behaviors file
    """

    def __init__(self, manager, file_directory):
        super().__init__(manager, file_directory)
        # get mapping from original cdd index to the one in faiss
        try:
            with open("data/recall/cddid2idx_recall.pkl", "rb") as f:
                self.cdd2index = pickle.load(f)
        except:
            manager.construct_cddidx_for_recall()
            with open("data/recall/cddid2idx_recall.pkl", "rb") as f:
                self.cdd2index = pickle.load(f)


    def init_behaviors(self):
        """
            init behavior logs given behaviors file.
        """
        # list of list of history news index
        histories = []
        # list of user index
        uindexes = []
        # list of impression indexes
        # self.impr_indexes = []
        impr_index = 0
        # only store positive behavior
        # list of lists, each list represents a
        imprs = []
        with open(self.behaviors_file, "r", encoding="utf-8") as rd:
            for idx in tqdm(rd, ncols=120, leave=True):
                _, uid, time, history, impr = idx.strip("\n").split("\t")

                history = [self.nid2index[i] for i in history.split()]

                impr_news = [self.nid2index[i.split("-")[0]] for i in impr.split()]
                labels = [int(i.split("-")[1]) for i in impr.split()]

                # user will always in uid2index
                uindex = self.uid2index[uid]

                pos_news = []
                for news, label in zip(impr_news, labels):
                    if label == 1:
                        pos_news.append(news)

                imprs.append((impr_index, pos_news))

                # 1 impression correspond to 1 of each of the following properties
                histories.append(history)
                uindexes.append(uindex)

                impr_index += 1

        save_dict = {
            "imprs": imprs,
            "histories": histories,
            "uindexes": uindexes
        }

        with open(self.behav_cache_path, "wb") as f:
            pickle.dump(save_dict, f)


    def __len__(self):
        """
            return length of the whole dataset
        """
        return len(self.imprs)


    def __getitem__(self,index):
        """ return data
        Args:
            index: the index for stored impression

        Returns:
            back_dic: dictionary of data slice
        """

        impr = self.imprs[index] # (impression_index, news_index)
        impr_index = impr[0]
        impr_news = impr[1]
        user_index = self.uindexes[impr_index]

        cdd_ids = impr_news

        # pad in his_id, not in histories
        his_ids = self.histories[impr_index][:self.his_size]
        # true means the corresponding history news is padded
        his_mask = np.zeros((self.his_size, 1))
        his_length = len(his_ids)
        if his_length == 0:
            his_mask[0] = 1
        else:
            his_mask[:his_length] = 1

        if self.descend_history:
            his_ids = his_ids[::-1] + [0] * (self.his_size - len(his_ids))
        else:
            his_ids = his_ids + [0] * (self.his_size - len(his_ids))

        cdd_encoded_index = self.encoded_news[cdd_ids]
        cdd_attn_mask = self.attn_mask[cdd_ids]
        his_encoded_index = self.encoded_news[his_ids]
        his_attn_mask = self.attn_mask[his_ids]

        cdd_ids = [self.cdd2index[x] for x in cdd_ids]

        back_dic = {
            "user_id": user_index,
            "cdd_id": np.asarray(cdd_ids),
            "his_id": np.asarray(his_ids),
            "cdd_encoded_index": cdd_encoded_index,
            "his_encoded_index": his_encoded_index,
            "cdd_attn_mask": cdd_attn_mask,
            "his_attn_mask": his_attn_mask,
            "his_mask": his_mask,
        }

        # word-level
        if self.subwords is not None:
            if self.reducer in ["bm25","entity","first"]:
                # subwords of history news don't accord with candidate news
                cdd_subword_index = self.subwords_original[cdd_ids]
                his_subword_index = self.subwords[his_ids]
            else:
                # matching
                cdd_subword_index = self.subwords[cdd_ids]
                his_subword_index = self.subwords[his_ids]
            back_dic["cdd_subword_index"] = cdd_subword_index
            back_dic["his_subword_index"] = his_subword_index

        if self.reducer == "matching" and hasattr(self, "attn_mask_dedup"):
            his_attn_mask_dedup = self.attn_mask_dedup[his_ids]
            back_dic["his_refined_mask"] = his_attn_mask_dedup

        elif self.reducer in ["bm25","entity","first"]:
            back_dic["cdd_encoded_index"] = self.encoded_news_original[cdd_ids]
            back_dic["cdd_attn_mask"] = self.attn_mask_original[cdd_ids]
            back_dic["his_attn_mask"] = back_dic["his_attn_mask"]

        elif self.reducer == "bow":
            back_dic["his_refined_mask"] = back_dic["his_attn_mask"]

        return back_dic
