import re
import os
import pickle
import logging

import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.utils import newsample, getId2idx

logger = logging.getLogger(__name__)


class MINDBaseDataset(Dataset):
    def __init__(self, manager, file_directory, news=True, behaviors=True) -> None:
        super().__init__()
        self.his_size = manager.his_size
        self.impr_size = manager.impr_size

        self.signal_length = manager.signal_length

        self.npratio = manager.npratio
        self.shuffle_pos = manager.shuffle_pos
        self.descend_history = manager.descend_history

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
                for k,v in behaviors.items():
                    setattr(self, k, v)

        if news:
            news_cache_directory = "/".join([cache_directory, "news", manager.get_bert_for_cache(), self.file_name])
            # used in manager.inspect()
            self.news_cache_directory = news_cache_directory + '/'
            self.news_cache_path = "/".join([news_cache_directory, "news.pkl"])
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
                    self.init_news()

            # synchronize all processes
            if manager.world_size > 1:
                dist.barrier()

            # logger.info("process NO.{} loading cached news tokenization from {}".format(manager.rank, self.news_cache_path))
            with open(self.news_cache_path, "rb") as f:
                news = pickle.load(f)
                self.encoded_news = news["encoded_news"][:, :self.signal_length]
                self.attn_mask = news["attn_mask"][:, :self.signal_length]

            # set the last token of a sequence to [SEP]
            sep_pos = self.encoded_news[:, -1] != self.pad_token_id
            self.encoded_news[:, -1] = self.sep_token_id * sep_pos


    def init_news(self):
        """
            parse news text into tokens, and save in the file, no assignment to self

            1. encode news text to tokens
            2. rerank words in the news text by bm25
            3. get subword indices
            4. get entities
        """
        # tokenize once, remove punctuations in BM25
        logger.info("tokenizing news...")
        with open(self.news_file, "r", encoding="utf-8") as rd:
            text_toks = []
            attention_masks = []
            padded_article = self.tokenizer("", padding='max_length', truncation=True, max_length=self.max_token_length)
            text_toks.append(padded_article['input_ids'])
            attention_masks.append(padded_article['attention_mask'])

            for idx in tqdm(rd, ncols=120, leave=True):
                nid, vert, subvert, title, ab, url, title_entity, abs_entity = idx.strip("\n").split("\t")

                article = " ".join([title, ab, subvert])
                token_ouput = self.tokenizer(article, padding='max_length', truncation=True, max_length=self.max_token_length)

                token_ids = token_ouput['input_ids']
                attn_mask = token_ouput['attention_mask']

                text_toks.append(token_ids)
                attention_masks.append(attn_mask)

            encoded_news = np.asarray(text_toks)
            attn_mask = np.asarray(attention_masks)

            with open(self.news_cache_directory + "news.pkl", "wb") as f:
                pickle.dump(
                    {
                        "encoded_news": encoded_news,
                        "attn_mask": attn_mask
                    },
                    f
                )


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



class MIND(MINDBaseDataset):
    def __init__(self, manager, file_directory):
        """ Map Style Dataset for MIND

        Args:
            manager(dict): pre-defined dictionary of hyper parameters
            file_directory(str): directory to news and behaviors file
        """
        super().__init__(manager, file_directory)


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

        return back_dic