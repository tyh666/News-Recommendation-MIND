# FIXME: need refactoring
class MIND_impr(Dataset):
    """ Map Style Dataset for MIND, return each impression once

    Args:
        manager(dict): pre-defined dictionary of hyper parameters
        news_file(str): path of news_file
        behaviors_file(str): path of behaviors_file
        shuffle(bool): whether to shuffle the order of impressions
    """

    def __init__(self, manager, news_file, behaviors_file, shuffle_pos=False, validate=False):
        # initiate the whole iterator
        self.npratio = manager.npratio
        self.shuffle_pos = manager.shuffle_pos

        self.news_file = news_file
        self.behaviors_file = behaviors_file
        self.title_length = manager.title_length
        self.abs_length = manager.abs_length
        self.his_size = manager.his_size

        self.multiview = manager.multiview
        self.k = manager.k

        # there are only two types of vocabulary
        self.vocab = getVocab("data/dictionaries/vocab.pkl")

        self.nid2index = getId2idx(
            "data/dictionaries/nid2idx_{}_{}.json".format(manager.scale, "dev"))
        self.uid2index = getId2idx(
            "data/dictionaries/uid2idx_{}.json".format(manager.scale))
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
            "user_id": user_index,
            "cdd_id": np.asarray(cdd_ids),
            "cdd_encoded_index": np.asarray(cdd_title_index),
            "his_id": np.asarray(his_ids),
            "his_encoded_index": his_title_index,
            "label": np.asarray(label)
        }

        return back_dic

# FIXME: need to update
class MIND_history(Dataset):
    """ Map Dataset for MIND, return each user's browsing history for encoding

    Args:
        manager
        news_file(str): path of news_file
        mode(str): train/test
    """

    def __init__(self, manager):
        reducer_map = {
            "none": "news.pkl",
            "matching": "news.pkl",
            "bm25": "bm25.pkl",
            "bow": "news.pkl",
            "entity": "entity.pkl",
            "first": "news.pkl"
        }
        # initiate the whole iterator
        self.npratio = manager.npratio
        self.shuffle_pos = manager.shuffle_pos
        self.signal_length = manager.signal_length
        self.his_size = manager.his_size
        self.impr_size = manager.impr_size
        self.k = manager.k
        self.descend_history = manager.descend_history
        self.reducer = manager.reducer
        self.granularity = manager.granularity

        self.mode = manager.get_mode_for_path()
        self.file_directory = manager.path + "MIND/"
        file_name = "MIND{}_{}/".format(manager.scale, self.mode)

        self.cache_directory = "/".join(["data/cache", manager.get_bert_for_cache(), file_name])
        self.news_path = self.cache_directory + reducer_map[self.reducer]
        self.behav_path = self.cache_directory + "{}/{}".format(self.impr_size, "behaviors.pkl")

        if not (os.path.exists(self.behav_path) and (os.path.exists(self.cache_directory + "news.pkl") and os.path.exists(self.cache_directory + "bm25.pkl") and os.path.exists(self.cache_directory + "entity.pkl"))):
            raise ValueError("please initialize MIND dataset before initializing MIND_history")

        logger.info("process NO.{} loading cached user behavior from {}".format(manager.rank, self.behav_path))
        with open(self.behav_path, "rb") as f:
            behaviors = pickle.load(f)
            self.histories = behaviors['histories']
            self.uindexes = behaviors['uindexes']

        logger.info("process NO.{} loading cached news tokenization from {}".format(manager.rank, self.news_path))
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

        refiner = None
        if manager.reducer == "matching":
            if not manager.no_dedup:
                from utils.utils import DeDuplicate
                refiner = DeDuplicate(manager)
        elif manager.reducer in ["bm25", "none", "entity", "first"]:
            from utils.utils import Truncate
            refiner = Truncate(manager)
        elif manager.reducer == "bow":
            from utils.utils import CountFreq
            refiner = CountFreq(manager)

        self.init_refinement(refiner)


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

        elif self.reducer in ["bow","none"]:
            self.encoded_news = refined_news
            self.attn_mask = refined_mask


    def __len__(self):
        """
            return length of the whole dataset
        """
        return len(self.uindexes)


    def __getitem__(self, idx):
        """ parse behavior log No.idx to training example

        Args:
            idx (int): impression index, start from zero

        Returns:
            dict of training data, including |npratio+1| cdd news word vector, |his_size+1| his news word vector etc.
        """
        user_index = self.uindexes[idx]
        his_ids = self.histories[idx][:self.his_size]
        his_mask = torch.zeros((self.his_size, 1))
        his_mask[:len(his_ids)] = 1

        if self.descend_history:
            his_ids = his_ids + [0] * (self.his_size - len(his_ids))
        else:
            his_ids = his_ids[::-1] + [0] * (self.his_size - len(his_ids))

        his_encoded_index = self.encoded_news[his_ids]
        his_attn_mask = self.attn_mask[his_ids]

        back_dic = {
            "user_id": user_index,
            "his_id": np.asarray(his_ids),
            "his_encoded_index": his_encoded_index,
            "his_attn_mask": his_attn_mask,
            "his_mask": his_mask,
        }

        if self.subwords is not None:
            if self.reducer in ["bm25","entity","first"]:
                his_subword_index = self.subwords[his_ids][:, :self.k + 1]
            else:
                his_subword_index = self.subwords[his_ids]
            back_dic["his_subword_index"] = his_subword_index

        if self.reducer == "matching":
            his_attn_mask_dedup = self.attn_mask_dedup[his_ids]
            back_dic["his_refined_mask"] = his_attn_mask_dedup

        elif self.reducer == "bow":
            back_dic["his_refined_mask"] = back_dic["his_attn_mask"]

        return back_dic