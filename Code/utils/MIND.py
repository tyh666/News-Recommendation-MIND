import re
import os
import pickle
import logging
import numpy as np
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
        # initiate the whole iterator
        self.npratio = config.npratio
        self.shuffle_pos = shuffle_pos
        self.signal_length = config.signal_length
        self.his_size = config.his_size
        self.impr_size = config.impr_size
        self.k = config.k
        pat = re.search('MIND/(.*_(.*)/)news', news_file)
        self.mode = pat.group(2)

        self.cache_path = '/'.join(['data/cache', config.embedding, pat.group(1), str(self.impr_size)+'/'])
        self.behav_path = re.search('(\w*)\.tsv', behaviors_file).group(1)

        if os.path.exists(self.cache_path + 'news.pkl'):
            with open(self.cache_path + 'news.pkl', 'rb') as f:
                news = pickle.load(f)
                for k,v in news.items():
                    setattr(self, k, v)

            with open(self.cache_path + 'behaviors.pkl', 'rb') as f:
                behaviors = pickle.load(f)
                for k,v in behaviors.items():
                    setattr(self, k, v)

        else:
            from transformers import BertTokenizerFast
            os.makedirs(self.cache_path, exist_ok=True)

            self.news_file = news_file
            self.behaviors_file = behaviors_file
            self.col_spliter = '\t'

            self.max_news_length = 512
            self.max_his_size = 100

            # there are only two types of vocabulary
            self.tokenizer = BertTokenizerFast.from_pretrained(config.bert, cache=config.path + 'bert_cache/')
            # self.tokenizer.max_model_input_sizes[config.bert] = 10000ok

            self.nid2index = getId2idx(
                'data/dictionaries/nid2idx_{}_{}.json'.format(config.scale, self.mode))
            self.uid2index = getId2idx(
                'data/dictionaries/uid2idx_{}.json'.format(config.scale))

            logger.info("encoding news and behaviors...")
            self.init_news()
            self.init_behaviors()


    def init_news(self):
        """
            init news information given news file, such as news_title_array.
        """

        # VERY IMPORTANT!!! FIXME
        # The nid2idx dictionary must follow the original order of news in news.tsv

        documents = ['[PAD]'*self.max_news_length]

        with open(self.news_file, "r", encoding='utf-8') as rd:
            for idx in rd:
                nid, vert, subvert, title, ab, url, _, _ = idx.strip("\n").split(self.col_spliter)
                # concat all fields to form the document
                # try:
                #     self.tokenizer.tokenize(' '.join(['[CLS]', title, ab, vert, subvert]))
                # except:
                #     print(' '.join(['[CLS]', title, ab, vert, subvert]))
                documents.append(' '.join(['[CLS]', title, ab, vert, subvert]))

        encoded_dict = self.tokenizer(documents, add_special_tokens=False, padding=True, truncation=True, max_length=self.max_news_length, return_tensors='np')
        self.encoded_news = encoded_dict.input_ids
        self.attn_mask = encoded_dict.attention_mask

        with open(self.cache_path + 'news.pkl', 'wb') as f:
            pickle.dump(
                {
                    'encoded_news': self.encoded_news,
                    'attn_mask': self.attn_mask
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
        # list of list of history padding length
        his_sizes = []
        # list of impression indexes
        # self.impr_indexes = []

        impr_index = 0

        # only store positive behavior
        if self.mode == 'train':
            # list of list of his cdd news index along with its impression index
            imprs = []
            # dictionary of list of unhis cdd news index
            negatives = {}

            with open(self.behaviors_file, "r", encoding='utf-8') as rd:
                for idx in rd:
                    _, uid, time, history, impr = idx.strip("\n").split(self.col_spliter)

                    history = [self.nid2index[i] for i in history.split()]
                    his_sizes.append(len(history))

                    # tailor user's history or pad 0
                    history = history[:self.max_his_size] + [0] * (self.max_his_size - len(history))
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
                    negatives[impr_index] = negative
                    uindexes.append(uindex)

                    impr_index += 1

            self.imprs = imprs
            self.histories = histories
            self.his_sizes = his_sizes
            self.negatives = negatives
            self.uindexes = uindexes

            save_dict = {
                'imprs': self.imprs,
                'histories': self.histories,
                'his_sizes': self.his_sizes,
                'negatives': self.negatives,
                'uindexes': self.uindexes
            }

        # store every behavior
        elif self.mode == 'dev':
            # list of every cdd news index along with its impression index and label
            imprs = []

            with open(self.behaviors_file, "r", encoding='utf-8') as rd:
                for idx in rd:
                    _, uid, time, history, impr = idx.strip("\n").split(self.col_spliter)

                    history = [self.nid2index[i] for i in history.split()]
                    his_sizes.append(len(history))
                    history = history[:self.max_his_size] + [0] * (self.max_his_size - len(history))

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
            self.his_sizes = his_sizes
            self.uindexes = uindexes

            save_dict = {
                'imprs': self.imprs,
                'histories': self.histories,
                'his_sizes': self.his_sizes,
                'uindexes': self.uindexes
            }

        # store every behavior
        elif self.mode == 'test':
            # list of every cdd news index along with its impression index and label
            imprs = []

            with open(self.behaviors_file, "r", encoding='utf-8') as rd:
                for idx in rd:
                    _, uid, time, history, impr = idx.strip("\n").split(self.col_spliter)

                    history = [self.nid2index[i] for i in history.split()]
                    his_sizes.append(len(history))
                    # tailor user's history or pad 0
                    history = history[:self.max_his_size] + [0] * (self.max_his_size - len(history))

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
            self.his_sizes = his_sizes
            self.uindexes = uindexes

            save_dict = {
                'imprs': self.imprs,
                'histories': self.histories,
                'his_sizes': self.his_sizes,
                'uindexes': self.uindexes
            }

        with open(self.cache_path + self.behav_path + '.pkl', 'wb') as f:
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


        user_index = [self.uindexes[impr_index]]

        # each time called to return positive one sample and its negative samples
        if self.mode == 'train':
            # user's unhis news in the same impression
            negs = self.negatives[impr_index]
            neg_list, neg_pad = newsample(negs, self.npratio)

            cdd_ids = [impr_news] + neg_list
            label = np.asarray([1] + [0] * self.npratio)

            if self.shuffle_pos:
                s = np.arange(0, len(label), 1)
                np.random.shuffle(s)
                cdd_ids = np.asarray(cdd_ids)[s]
                label = np.asarray(label)[s]

            label = np.arange(0, len(cdd_ids), 1)[label == 1][0]

            his_ids = self.histories[impr_index][:self.his_size]

            # true means the corresponding history news is padded
            his_mask = np.zeros((self.his_size), dtype=bool)
            his_mask[:self.his_sizes[impr_index]] = 1

            # pad in cdd
            # cdd_mask = [1] * neg_pad + [0] * (self.npratio + 1 - neg_pad)

            cdd_encoded_index = self.encoded_news[cdd_ids][:, :self.signal_length]
            his_encoded_index = self.encoded_news[his_ids][:, :self.signal_length]
            cdd_attn_mask = self.attn_mask[cdd_ids][:, :self.signal_length]
            his_attn_mask = self.attn_mask[his_ids][:, :self.signal_length]

            back_dic = {
                "user_index": np.asarray(user_index),
                # "cdd_mask": np.asarray(neg_pad),
                'cdd_id': np.asarray(cdd_ids),
                'his_id': np.asarray(his_ids),
                "cdd_encoded_index": cdd_encoded_index,
                "his_encoded_index": his_encoded_index,
                "cdd_attn_mask": cdd_attn_mask,
                "his_attn_mask": his_attn_mask,
                "his_mask": his_mask,
                "label": label
            }

            return back_dic

        # each time called return one sample, and no labels
        elif self.mode == 'dev':
            cdd_ids = impr_news

            his_ids = self.histories[impr_index][:self.his_size]

            user_index = [self.uindexes[impr_index]]
            label = impr[2]

            # true means the corresponding history news is padded
            his_mask = np.zeros((self.his_size), dtype=bool)
            his_mask[:self.his_sizes[impr_index]] = 1

            cdd_encoded_index = self.encoded_news[cdd_ids][:, :self.signal_length]
            his_encoded_index = self.encoded_news[his_ids][:, :self.signal_length]
            cdd_attn_mask = self.attn_mask[cdd_ids][:, :self.signal_length]
            his_attn_mask = self.attn_mask[his_ids][:, :self.signal_length]

            back_dic = {
                "impression_index": impr_index + 1,
                "user_index": np.asarray(user_index),
                'cdd_id': np.asarray(cdd_ids),
                'his_id': np.asarray(his_ids),
                "cdd_encoded_index": cdd_encoded_index,
                "his_encoded_index": his_encoded_index,
                "cdd_attn_mask": cdd_attn_mask,
                "his_attn_mask": his_attn_mask,
                "his_mask": his_mask,
                "label": np.asarray(label)
            }
            return back_dic

        elif self.mode == 'test':
            cdd_ids = impr_news

            his_ids = self.histories[impr_index][:self.his_size]

            user_index = [self.uindexes[impr_index]]
            # true means the corresponding history news is padded
            his_mask = np.zeros((self.his_size), dtype=bool)
            his_mask[:self.his_sizes[impr_index]] = 1

            cdd_encoded_index = self.encoded_news[cdd_ids][:, :self.signal_length]
            his_encoded_index = self.encoded_news[his_ids][:, :self.signal_length]
            cdd_attn_mask = self.attn_mask[cdd_ids][:, :self.signal_length]
            his_attn_mask = self.attn_mask[his_ids][:, :self.signal_length]

            back_dic = {
                "impression_index": impr_index + 1,
                "user_index": np.asarray(user_index),
                'cdd_id': np.asarray(cdd_ids),
                'his_id': np.asarray(his_ids),
                "cdd_encoded_index": cdd_encoded_index,
                "his_encoded_index": his_encoded_index,
                "cdd_attn_mask": cdd_attn_mask,
                "his_attn_mask": his_attn_mask,
                "his_mask": his_mask
            }
            return back_dic

        else:
            raise ValueError("Mode {} not defined".format(self.mode))


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
        pat = re.search('MIND/(.*_(.*)/)news', news_file)
        self.mode = pat.group(2)

        self.cache_path = '/'.join(['data/cache', config.embedding, pat.group(1), str(self.impr_size)+'/'])

        if os.path.exists(self.cache_path + 'news.pkl'):
            with open(self.cache_path + 'news.pkl', 'rb') as f:
                news = pickle.load(f)
                for k,v in news.items():
                    setattr(self, k, v)

        else:
            from transformers import BertTokenizerFast
            os.makedirs(self.cache_path, exist_ok=True)

            self.news_file = news_file
            self.behaviors_file = behaviors_file
            self.col_spliter = '\t'

            self.max_news_length = 512

            # there are only two types of vocabulary
            self.tokenizer = BertTokenizerFast.from_pretrained(config.bert, cache=config.path + 'bert_cache/')

            self.nid2index = getId2idx(
                'data/dictionaries/nid2idx_{}_{}.json'.format(config.scale, self.mode))

            logger.info("encoding news...")
            self.init_news()


    def init_news(self):
        """
            init news information given news file, such as news_title_array.
        """

        # VERY IMPORTANT!!! FIXME
        # The nid2idx dictionary must follow the original order of news in news.tsv

        documents = ['[PAD]'*(self.max_news_length - 1)]

        with open(self.news_file, "r", encoding='utf-8') as rd:
            for idx in rd:
                nid, vert, subvert, title, ab, url, _, _ = idx.strip("\n").split(self.col_spliter)
                # concat all fields to form the document
                # try:
                #     self.tokenizer.tokenize(' '.join(['[CLS]', title, ab, vert, subvert]))
                # except:
                #     print(' '.join(['[CLS]', title, ab, vert, subvert]))
                documents.append(' '.join(['[CLS]', title, ab, vert, subvert]))

        encoded_dict = self.tokenizer(documents, add_special_tokens=False, padding=True, truncation=True, max_length=self.max_news_length, return_tensors='np')
        self.encoded_news = encoded_dict.input_ids
        self.attn_mask = encoded_dict.attention_mask

        with open(self.cache_path + 'news.pkl', 'wb') as f:
            pickle.dump(
                {
                    'encoded_news': self.encoded_news,
                    'attn_mask': self.attn_mask
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
        self.col_spliter = '\t'
        self.title_length = config.title_length
        self.abs_length = config.abs_length
        self.his_size = config.his_size

        self.multiview = config.multiview
        self.k = config.k

        # there are only two types of vocabulary
        self.vocab = getVocab('data/dictionaries/vocab.pkl')

        self.nid2index = getId2idx(
            'data/dictionaries/nid2idx_{}_{}.json'.format(config.scale, 'dev'))
        self.uid2index = getId2idx(
            'data/dictionaries/uid2idx_{}.json'.format(config.scale))
        self.vert2onehot = getId2idx(
            'data/dictionaries/vert2onehot.json'
        )
        self.subvert2onehot = getId2idx(
            'data/dictionaries/subvert2onehot.json'
        )

        self.mode = 'dev'

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

        with open(self.news_file, "r", encoding='utf-8') as rd:
            for idx in rd:
                nid, vert, subvert, title, ab, url, _, _ = idx.strip("\n").split(self.col_spliter)

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

        with open(self.behaviors_file, "r", encoding='utf-8') as rd:
            for idx in rd:
                _, uid, time, history, impr = idx.strip("\n").split(self.col_spliter)

                history = [self.nid2index[i] for i in history.split()]
                # tailor user's history or pad 0
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
            "impression_index": impr_index + 1,
            "user_index": np.asarray(user_index),
            'cdd_id': np.asarray(cdd_ids),
            "cdd_encoded_index": np.asarray(cdd_title_index),
            'his_id': np.asarray(his_ids),
            "his_encoded_index": his_title_index,
            "label": np.asarray(label)
        }

        return back_dic