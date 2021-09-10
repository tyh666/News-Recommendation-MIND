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

class MIND(Dataset):
    """ Map Style Dataset for MIND, use bert tokenizer

    Args:
        config(dict): pre-defined dictionary of hyper parameters
        news_file(str): path of news_file
        behaviors_file(str): path of behaviors_file
        shuffle(bool): whether to shuffle the order of impressions
    """

    def __init__(self, config, news_file, behaviors_file, shuffle_pos=False):
        reducer_map = {
            "matching": "news.pkl",
            "bm25": "bm25.pkl",
            "bow": "news.pkl",
            "entity": "entity.pkl",
            "first": "news.pkl"
        }
        # initiate the whole iterator
        self.npratio = config.npratio
        self.shuffle_pos = shuffle_pos
        self.signal_length = config.signal_length
        self.his_size = config.his_size
        self.impr_size = config.impr_size
        self.k = config.k
        self.ascend_history = config.ascend_history
        self.reducer = config.reducer
        self.granularity = config.granularity

        pat = re.search("MIND/(.*_(.*)/)news", news_file)
        self.mode = pat.group(2)

        self.cache_directory = "/".join(["data/cache", config.embedding, pat.group(1)])
        self.news_path = self.cache_directory + reducer_map[self.reducer]
        self.behav_path = self.cache_directory + "{}/{}".format(self.impr_size, re.search("(\w*\.)tsv", behaviors_file).group(1) + ".pkl")

        # only preprocess on the master node, the worker can directly load the cache
        if config.rank in [-1, 0]:
            if not os.path.exists(self.behav_path):
                logger.info("encoding user behaviors of {}...".format(behaviors_file))
                os.makedirs(self.cache_directory + str(self.impr_size), exist_ok=True)
                self.behaviors_file = behaviors_file
                try:
                    # VERY IMPORTANT!!!
                    # The nid2idx dictionary must follow the original order of news in news.tsv
                    self.nid2index = getId2idx("data/dictionaries/nid2idx_{}_{}.json".format(config.scale, self.mode))
                except FileNotFoundError:
                    config.construct_nid2idx()
                    self.nid2index = getId2idx("data/dictionaries/nid2idx_{}_{}.json".format(config.scale, self.mode))
                try:
                    self.uid2index = getId2idx("data/dictionaries/uid2idx_{}.json".format(config.scale))
                except FileNotFoundError:
                    config.construct_uid2idx()
                    self.uid2index = getId2idx("data/dictionaries/uid2idx_{}.json".format(config.scale))

                self.init_behaviors()


            if not (os.path.exists(self.cache_directory + "news.pkl") and os.path.exists(self.cache_directory + "bm25.pkl") and os.path.exists(self.cache_directory + "entity.pkl")):
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(config.bert, cache=config.path + "bert_cache/")

                logger.info("encoding news of {}...".format(news_file))
                self.news_file = news_file
                self.embedding = config.embedding

                self.max_token_length = 512
                self.max_reduction_length = 30

                self.nid2index = getId2idx("data/dictionaries/nid2idx_{}_{}.json".format(config.scale, self.mode))

                self.convert_tokens_to_words = config.convert_tokens_to_words

                self.init_news()

        # synchronize all processes
        if config.world_size > 1:
            dist.barrier()

        logger.info("process NO.{} loading cached user behavior from {}".format(config.rank, self.behav_path))
        with open(self.behav_path, "rb") as f:
            behaviors = pickle.load(f)
            for k,v in behaviors.items():
                setattr(self, k, v)

        logger.info("process NO.{} loading cached news tokenization from {}".format(config.rank, self.news_path))
        with open(self.news_path, "rb") as f:
            news = pickle.load(f)
            self.encoded_news = news["encoded_news"]
            self.attn_mask = news["attn_mask"]
            if self.granularity in ["avg","sum"]:
                self.subwords = news["subwords_all"][:, :self.signal_length]
            elif self.granularity == "first":
                self.subwords = news["subwords_first"][:, :self.signal_length]
            else:
                self.subwords = None

        if self.reducer in ["bm25", "entity", "first"]:
            with open(self.cache_directory + "news.pkl", "rb") as f:
                news = pickle.load(f)
                self.encoded_news_original = news["encoded_news"]
                self.attn_mask_original = news["attn_mask"]
                if self.granularity in ["avg","sum"]:
                    self.subwords_original = news["subwords_all"][:, :self.signal_length]
                elif self.granularity == "first":
                    self.subwords_original = news["subwords_first"][:, :self.signal_length]
                else:
                    self.subwords_original = None

        if config.reducer == "matching":
            if not config.no_dedup:
                from utils.utils import DeDuplicate
                refiner = DeDuplicate(config)
        elif config.reducer in ["bm25", "entity", "first"]:
            from utils.utils import Truncate
            refiner = Truncate(config)
        elif config.reducer == "bow":
            from utils.utils import CountFreq
            refiner = CountFreq(config)
        else:
            refiner = None

        self.init_refinement(refiner)


    def init_news(self):
        """
            parse news text into tokens, and save in the file, no assignment to self

            1. encode news text to tokens
            2. rerank words in the news text by bm25
            3. get subword indices
            4. get entities
        """
        articles = [""]
        entities = [""]
        with open(self.news_file, "r", encoding="utf-8") as rd:
            for idx in tqdm(rd):
                nid, vert, subvert, title, ab, url, title_entity, abs_entity = idx.strip("\n").split("\t")
                article = " ".join([title, ab, vert, subvert])
                article = re.sub("\'|\"", '', article)
                tokens = self.tokenizer.tokenize(article)[:self.max_token_length]
                # unify subwords
                words = self.convert_tokens_to_words(tokens)
                articles.append(' '.join(words))

                entity_dic = dict()
                title_entity = json.loads(title_entity)
                abs_entity = json.loads(abs_entity)
                for entity in title_entity + abs_entity:
                    surface_forms = entity["SurfaceForms"]
                    if len(surface_forms):
                        entity_dic[surface_forms[0].lower()] = 1

                if len(entity_dic) == 0:
                    entities.append(' '.join(words[:self.max_reduction_length]))
                else:
                    entities.append(' '.join(list(entity_dic.keys())))

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
            for text in tqdm(texts):
                tokens = tokenizer.tokenize(text)

                # maintain subword entry
                subword_all = [[0,0]]
                # mask subword entry
                subword_first = [[0,0]]

                i = 0
                j = 0
                for token in tokens:
                    if token.startswith("##"):
                        j += 1
                        # subword.append([0,0])
                        subword_all.append([i,j])
                        subword_first.append([0,0])

                    else:
                        i += 1
                        j += 1
                        subword_all.append([i,j])
                        subword_first.append([i,j])

                token_ids = tokenizer.convert_tokens_to_ids(["[CLS]"]) + tokenizer.convert_tokens_to_ids(tokens[:max_length - 1])
                pad_length = max_length - len(token_ids)

                text_toks.append(token_ids + [0] * pad_length)
                attention_masks.append([1] * min(len(token_ids), max_length) + [0] * pad_length)

                subword_all.extend([[0,0]] * pad_length)
                subword_first.extend([[0,0]] * pad_length)
                subwords_all.append(subword_all[:max_length])
                subwords_first.append(subword_first[:max_length])

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

        def parse_texts_deberta(tokenizer, texts, news_path, max_length):
            """
            convert texts to tokens indices and get subword indices
            """
            text_toks = []
            attention_masks = []
            subwords_all = []
            subwords_first = []
            for text in texts:
                tokens = tokenizer.tokenize(text)

                # maintain subword entry
                subword_all = [[0,0]]
                # mask subword entry
                subword_first = [[0,0]]

                i = 0
                j = 0
                for index,token in enumerate(tokens):
                    # not subword
                    if index == 0 or token.startswith("Ġ") or token in r"[.&*()+=/\<>,!?;:~`@#$%^]":
                        i += 1
                        j += 1
                        subword_all.append([i,j])
                        subword_first.append([i,j])

                    # subword
                    else:
                        j += 1
                        # subword.append([0,0])
                        subword_all.append([i,j])
                        subword_first.append([0,0])

                token_ids = tokenizer.convert_tokens_to_ids(["[CLS]"]) + tokenizer.convert_tokens_to_ids(tokens[:max_length - 1])
                pad_length = max_length - len(token_ids)

                text_toks.append(token_ids + [0] * pad_length)
                attention_masks.append([1] * min(len(token_ids), max_length) + [0] * pad_length)

                subword_all.extend([[0,0]] * pad_length)
                subword_first.extend([[0,0]] * pad_length)
                subwords_all.append(subword_all[:max_length])
                subwords_first.append(subword_first[:max_length])

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

        if self.embedding == 'bert':
            logger.info("tokenizing news...")
            parse_texts_bert(self.tokenizer, articles, self.cache_directory + "news.pkl", self.max_token_length)
            logger.info("tokenizing bm25 ordered news...")
            parse_texts_bert(self.tokenizer, articles_bm25, self.cache_directory + "bm25.pkl", self.max_reduction_length)
            logger.info("tokenizing entities...")
            parse_texts_bert(self.tokenizer, entities, self.cache_directory + "entity.pkl", self.max_reduction_length)

        elif self.embedding == 'deberta':
            logger.info("tokenizing news...")
            parse_texts_deberta(self.tokenizer, articles, self.cache_directory + "news.pkl", self.max_token_length)
            logger.info("tokenizing bm25 ordered news...")
            parse_texts_deberta(self.tokenizer, articles_bm25, self.cache_directory + "bm25.pkl", self.max_reduction_length)
            logger.info("tokenizing entities...")
            parse_texts_deberta(self.tokenizer, entities, self.cache_directory + "entity.pkl", self.max_reduction_length)


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
                for idx in tqdm(rd):
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

            self.imprs = imprs
            self.histories = histories
            self.negatives = negatives
            self.uindexes = uindexes

            save_dict = {
                "imprs": self.imprs,
                "histories": self.histories,
                "negatives": self.negatives,
                "uindexes": self.uindexes
            }

        # store every behavior
        elif self.mode == "dev":
            # list of every cdd news index along with its impression index and label
            imprs = []

            with open(self.behaviors_file, "r", encoding="utf-8") as rd:
                for idx in rd:
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

            self.imprs = imprs
            self.histories = histories
            self.uindexes = uindexes

            save_dict = {
                "imprs": self.imprs,
                "histories": self.histories,
                "uindexes": self.uindexes
            }

        # store every behavior
        elif self.mode == "test":
            # list of every cdd news index along with its impression index and label
            imprs = []

            with open(self.behaviors_file, "r", encoding="utf-8") as rd:
                for idx in rd:
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

            self.imprs = imprs
            self.histories = histories
            self.uindexes = uindexes

            save_dict = {
                "imprs": self.imprs,
                "histories": self.histories,
                "uindexes": self.uindexes
            }

        with open(self.behav_path, "wb") as f:
            pickle.dump(save_dict, f)


    def init_refinement(self, refiner):
        """
            token level refinement, determined by reducer

            matching -> deduplicate
            bm25 -> truncate
            bow -> count
        """
        if not refiner:
            return

        refined_news, refined_mask = refiner(self.encoded_news, self.attn_mask)
        if self.reducer == "matching":
            self.encoded_news = refined_news
            self.attn_mask_dedup = refined_mask
            # truncate the attention mask
            self.attn_mask = self.attn_mask[:, :self.signal_length]

        elif self.reducer in ["bm25", "entity", "first"]:
            self.encoded_news = refined_news
            self.attn_mask = refined_mask
            # truncate the original text tokens
            self.encoded_news_original = self.encoded_news_original[:, :self.signal_length]
            self.attn_mask_original = self.attn_mask_original[:, :self.signal_length]

        elif self.reducer == "bow":
            self.encoded_news = refined_news
            self.attn_mask = refined_mask

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

        user_index = [self.uindexes[impr_index]]

        # each time called to return positive one sample and its negative samples
        if self.mode == "train":
            # user"s unhis news in the same impression
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

            his_ids = self.histories[impr_index][:self.his_size]

            cdd_mask = torch.zeros((cdd_size, 1))
            cdd_mask[:neg_num + 1] = 1

            # true means the corresponding history news is padded
            his_mask = torch.zeros((self.his_size, 1), dtype=bool)
            his_mask[:len(his_ids)] = 1

            if self.ascend_history:
                his_ids = his_ids + [0] * (self.his_size - len(his_ids))
            else:
                his_ids = his_ids[::-1] + [0] * (self.his_size - len(his_ids))

            cdd_encoded_index = self.encoded_news[cdd_ids]
            cdd_attn_mask = self.attn_mask[cdd_ids]
            his_encoded_index = self.encoded_news[his_ids]
            his_attn_mask = self.attn_mask[his_ids]

            back_dic = {
                "user_index": np.asarray(user_index),
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

            if self.subwords is not None:
                if self.reducer in ["bm25","entity","first"]:
                    cdd_subword_index = self.subwords_original[cdd_ids]
                    his_subword_index = self.subwords[his_ids][:, :self.k + 1]
                else:
                    cdd_subword_index = self.subwords[cdd_ids]
                    his_subword_index = self.subwords[his_ids]
                back_dic["cdd_subword_index"] = cdd_subword_index
                back_dic["his_subword_index"] = his_subword_index

            if self.reducer == "matching":
                his_attn_mask_dedup = self.attn_mask_dedup[his_ids]
                back_dic["his_refined_mask"] = his_attn_mask_dedup

            elif self.reducer in ["bm25","entity","first"]:
                back_dic["cdd_encoded_index"] = self.encoded_news_original[cdd_ids]
                back_dic["cdd_attn_mask"] = self.attn_mask_original[cdd_ids]

            elif self.reducer == "bow":
                back_dic["his_refined_mask"] = back_dic["his_attn_mask"]

            return back_dic

        # each time called return one sample, and no labels
        elif self.mode == "dev":
            cdd_ids = impr_news
            cdd_size = len(cdd_ids)

            his_ids = self.histories[impr_index][:self.his_size]
            # true means the corresponding history news is padded
            his_mask = torch.zeros((self.his_size, 1), dtype=bool)
            his_mask[:len(his_ids)] = 1

            if self.ascend_history:
                his_ids = his_ids + [0] * (self.his_size - len(his_ids))
            else:
                his_ids = his_ids[::-1] + [0] * (self.his_size - len(his_ids))

            user_index = [self.uindexes[impr_index]]
            label = impr[2]

            cdd_encoded_index = self.encoded_news[cdd_ids]
            cdd_attn_mask = self.attn_mask[cdd_ids]
            his_encoded_index = self.encoded_news[his_ids]
            his_attn_mask = self.attn_mask[his_ids]

            back_dic = {
                "impr_index": impr_index + 1,
                "user_index": np.asarray(user_index),
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
                    his_subword_index = self.subwords[his_ids][:, :self.k + 1]
                else:
                    cdd_subword_index = self.subwords[cdd_ids]
                    his_subword_index = self.subwords[his_ids]
                back_dic["cdd_subword_index"] = cdd_subword_index
                back_dic["his_subword_index"] = his_subword_index

            if self.reducer == "matching":
                his_attn_mask_dedup = self.attn_mask_dedup[his_ids]
                back_dic["his_refined_mask"] = his_attn_mask_dedup

            elif self.reducer in ["bm25","entity","first"]:
                back_dic["cdd_encoded_index"] = self.encoded_news_original[cdd_ids]
                back_dic["cdd_attn_mask"] = self.attn_mask_original[cdd_ids]

            elif self.reducer == "bow":
                back_dic["his_refined_mask"] = back_dic["his_attn_mask"]

            return back_dic

        elif self.mode == "test":
            cdd_ids = impr_news
            cdd_size = len(cdd_ids)

            his_ids = self.histories[impr_index][:self.his_size]
            # true means the corresponding history news is padded
            his_mask = torch.zeros((self.his_size, 1), dtype=bool)
            his_mask[:len(his_ids)] = 1

            if self.ascend_history:
                his_ids = his_ids + [0] * (self.his_size - len(his_ids))
            else:
                his_ids = his_ids[::-1] + [0] * (self.his_size - len(his_ids))

            user_index = [self.uindexes[impr_index]]

            cdd_encoded_index = self.encoded_news[cdd_ids]
            cdd_attn_mask = self.attn_mask[cdd_ids]
            his_encoded_index = self.encoded_news[his_ids]
            his_attn_mask = self.attn_mask[his_ids]

            back_dic = {
                "impr_index": impr_index + 1,
                "user_index": np.asarray(user_index),
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
                    his_subword_index = self.subwords[his_ids][:, :self.k + 1]
                else:
                    cdd_subword_index = self.subwords[cdd_ids]
                    his_subword_index = self.subwords[his_ids]
                back_dic["cdd_subword_index"] = cdd_subword_index
                back_dic["his_subword_index"] = his_subword_index

            if self.reducer == "matching":
                his_attn_mask_dedup = self.attn_mask_dedup[his_ids]
                back_dic["his_refined_mask"] = his_attn_mask_dedup

            elif self.reducer in ["bm25","entity","first"]:
                back_dic["cdd_encoded_index"] = self.encoded_news_original[cdd_ids]
                back_dic["cdd_attn_mask"] = self.attn_mask_original[cdd_ids]

            elif self.reducer == "bow":
                back_dic["his_refined_mask"] = back_dic["his_attn_mask"]

            return back_dic

        else:
            raise ValueError("Mode {} not defined".format(self.mode))


# FIXME: refactor with bm25
class MIND_news(Dataset):
    """ Map Dataset for MIND, return each news, intended for pipeline(obtaining news representation in advance)

    Args:
        config(dict): pre-defined dictionary of hyper parameters
        news_file(str): path of news_file
        mode(str): train/test
    """

    def __init__(self, config, news_file, behaviors_file, shuffle_pos=False):
        # initiate the whole iterator
        self.shuffle_pos = shuffle_pos
        self.signal_length = config.signal_length
        self.k = config.k
        pat = re.search("MIND/(.*_(.*)/)news", news_file)
        self.mode = pat.group(2)

        self.news_path = "/".join(["data/cache", config.embedding, pat.group(1), "news.pkl"])

        if os.path.exists(self.news_path):
            with open(self.news_path, "rb") as f:
                news = pickle.load(f)
                for k,v in news.items():
                    setattr(self, k, v)

        else:
            from transformers import BertTokenizerFast
            os.makedirs(self.cache_directory, exist_ok=True)

            self.news_file = news_file
            self.behaviors_file = behaviors_file

            self.max_news_length = 512

            # there are only two types of vocabulary
            self.tokenizer = BertTokenizerFast.from_pretrained(config.bert, cache=config.path + "bert_cache/")

            self.nid2index = getId2idx(
                "data/dictionaries/nid2idx_{}_{}.json".format(config.scale, self.mode))

            logger.info("encoding news...")
            self.init_news()


    def init_news(self):
        """
            init news information given news file, such as news_title_array.
        """

        # VERY IMPORTANT!!! FIXME
        # The nid2idx dictionary must follow the original order of news in news.tsv

        documents = ["[PAD]"*(self.max_news_length - 1)]

        with open(self.news_file, "r", encoding="utf-8") as rd:
            for idx in rd:
                nid, vert, subvert, title, ab, url, _, _ = idx.strip("\n").split("\t")
                # concat all fields to form the document
                # try:
                #     self.tokenizer.tokenize(" ".join(["[CLS]", title, ab, vert, subvert]))
                # except:
                #     print(" ".join(["[CLS]", title, ab, vert, subvert]))
                documents.append(" ".join(["[CLS]", title, ab, vert, subvert]))

        encoded_dict = self.tokenizer(documents, add_special_tokens=False, padding=True, truncation=True, max_length=self.max_news_length, return_tensors="np")
        self.encoded_news = encoded_dict.input_ids
        self.attn_mask = encoded_dict.attention_mask

        with open(self.news_path, "wb") as f:
            pickle.dump(
                {
                    "encoded_news": self.encoded_news,
                    "attn_mask": self.attn_mask
                },
                f
            )


    def __len__(self):
        return len(self.news_title_array)


    def __getitem__(self, idx):
        """ parse behavior log No.idx to training example

        Args:
            idx (int): impression index, start from zero

        Returns:
            dict of training data, including |npratio+1| cdd news word vector, |his_size+1| his news word vector etc.
        """
        encoded_index = self.encoded_news[idx][:self.signal_length]
        attn_mask = self.attn_mask[idx][:self.signal_length]
        return {
            "cdd_id": np.asarray([idx]),
            "cdd_encoded_index": encoded_index,
            "cdd_attn_mask": attn_mask
        }


# FIXME: need refactoring
class MIND_impr(Dataset):
    """ Map Style Dataset for MIND, return each impression once

    Args:
        config(dict): pre-defined dictionary of hyper parameters
        news_file(str): path of news_file
        behaviors_file(str): path of behaviors_file
        shuffle(bool): whether to shuffle the order of impressions
    """

    def __init__(self, config, news_file, behaviors_file, shuffle_pos=False, validate=False):
        # initiate the whole iterator
        self.npratio = config.npratio
        self.shuffle_pos = shuffle_pos

        self.news_file = news_file
        self.behaviors_file = behaviors_file
        self.title_length = config.title_length
        self.abs_length = config.abs_length
        self.his_size = config.his_size

        self.multiview = config.multiview
        self.k = config.k

        # there are only two types of vocabulary
        self.vocab = getVocab("data/dictionaries/vocab.pkl")

        self.nid2index = getId2idx(
            "data/dictionaries/nid2idx_{}_{}.json".format(config.scale, "dev"))
        self.uid2index = getId2idx(
            "data/dictionaries/uid2idx_{}.json".format(config.scale))
        self.vert2onehot = getId2idx(
            "data/dictionaries/vert2onehot.json"
        )
        self.subvert2onehot = getId2idx(
            "data/dictionaries/subvert2onehot.json"
        )

        self.mode = "dev"

        self.init_news()
        self.init_behaviors()

    def init_news(self):
        """
            init news information given news file, such as news_title_array.
        """

        # VERY IMPORTANT!!! FIXME
        # The nid2idx dictionary must follow the original order of news in news.tsv

        titles = [[1]*self.max_title_length]
        title_lengths = [0]

        with open(self.news_file, "r", encoding="utf-8") as rd:
            for idx in rd:
                nid, vert, subvert, title, ab, url, _, _ = idx.strip("\n").split("\t")

                title_token = tokenize(title, self.vocab)
                titles.append(title_token[:self.max_title_length] + [1] * (self.max_title_length - len(title_token)))
                title_lengths.append(len(title_token))

        # self.titles = titles
        self.news_title_array = np.asarray(titles)
        self.title_lengths = np.asarray(title_lengths)

    def init_behaviors(self):
        """
            init behavior logs given behaviors file.
        """
        # list of list of history news index
        histories = []
        # list of user index
        uindexes = []
        # list of list of history padding length
        his_sizes = []
        # list of impression indexes
        # self.impr_indexes = []

        # list of every cdd news index along with its impression index and label
        imprs = []

        impr_index = 0

        with open(self.behaviors_file, "r", encoding="utf-8") as rd:
            for idx in rd:
                _, uid, time, history, impr = idx.strip("\n").split("\t")

                history = [self.nid2index[i] for i in history.split()]
                # tailor user"s history or pad 0
                history = history[:self.max_his_size] + [0] * (self.max_his_size - len(history))
                impr_news = [self.nid2index[i.split("-")[0]] for i in impr.split()]
                labels = [int(i.split("-")[1]) for i in impr.split()]
                # user will always in uid2index
                uindex = self.uid2index[uid]

                # store every impression
                imprs.append((impr_index, impr_news[0], labels[0]))

                # 1 impression correspond to 1 of each of the following properties
                histories.append(history)
                uindexes.append(uindex)

                impr_index += 1

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


        user_index = [self.uindexes[impr_index]]

        cdd_ids = [impr_news]

        his_ids = self.histories[impr_index][:self.his_size]

        user_index = [self.uindexes[impr_index]]
        label = impr[2]

        cdd_title_index = self.news_title_array[cdd_ids][:, :self.title_length]
        his_title_index = self.news_title_array[his_ids][:, :self.title_length]
        back_dic = {
            "impr_index": impr_index + 1,
            "user_index": np.asarray(user_index),
            "cdd_id": np.asarray(cdd_ids),
            "cdd_encoded_index": np.asarray(cdd_title_index),
            "his_id": np.asarray(his_ids),
            "his_encoded_index": his_title_index,
            "label": np.asarray(label)
        }

        return back_dic