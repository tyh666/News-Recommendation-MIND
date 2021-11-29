import torch
import re
import os
import logging
import argparse
import json
import pickle
import time
import random
import smtplib
import pandas as pd
import numpy as np
import scipy.stats as ss
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import sampler

from datetime import datetime, timedelta
from tqdm.auto import tqdm
from typing import OrderedDict
from collections import defaultdict
from transformers import get_linear_schedule_with_warmup
# from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, accuracy_score, f1_score
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .utils import _group_lists

logger = logging.getLogger(__name__)

hparam_list = ["epochs", "device", "path", "title_length", "step", "checkpoint", "smoothing", "num_workers", "pin_memory", "interval", "npratio", "metrics", "aggregator", "head_num", "rank", "unilm_path", "unilm_config_path"]

class Manager():
    """
    wrap training/evaluating processes
    """
    def __init__(self, config=None):
        """
            customize hyper parameters in command line
        """
        if config is None:
            parser = argparse.ArgumentParser()
            parser.add_argument("-s", "--scale", dest="scale", help="data scale", choices=["demo", "small", "large", "whole"], default="large")
            parser.add_argument("-m", "--mode", dest="mode", help="train or test", choices=["train", "dev", "test", "encode", "inspect", "analyse", "recall"], default="train")
            parser.add_argument("-e", "--epochs", dest="epochs", help="epochs to train the model", type=int, default=10)
            parser.add_argument("-d","--device", dest="device", help="device to run on, -1 means cpu", choices=[i for i in range(-1,10)], type=int, default=0)
            parser.add_argument("-p", "--path", dest="path", type=str, default="../../Data/", help="root path for large-scale reusable data")
            parser.add_argument("-f", "--fast", dest="fast", help="enable fast evaluation/test", default=True)
            parser.add_argument("-n", "--news", dest="news", help="which news to inspect", type=str, default=None)
            parser.add_argument("-c", "--case", dest="case", help="whether to return the sample for case study", action="store_true", default=False)
            parser.add_argument("-rt", "--recall_type", dest="recall_type", help="recall type", choices=["s","d","sd"], default=None)
            parser.add_argument("-it", "--inspect_type", dest="inspect_type", help="the dataset to inspect", choices=["dev","test","all"], default=None)

            parser.add_argument("-bs", "--batch_size", dest="batch_size", help="batch size in training", type=int, default=32)
            parser.add_argument("-bsn", "--batch_size_news", dest="batch_size_news", help="batch size of loader_news", type=int, default=500)
            parser.add_argument("-hs", "--his_size", dest="his_size",help="history size", type=int, default=50)
            parser.add_argument("-is", "--impr_size", dest="impr_size", help="impression size for evaluating", type=int, default=2000)
            parser.add_argument("-sl", "--signal_length", dest="signal_length", help="length of the bert tokenized tokens", type=int, default=30)

            parser.add_argument("-hd", "--hidden_dim", dest="hidden_dim", help="number of hidden states", type=int, default=150)
            parser.add_argument("-ed", "--embedding_dim", dest="embedding_dim", help="number of embedding states", type=int, default=768)
            parser.add_argument("-bd", "--bert_dim", dest="bert_dim", help="number of hidden states in pre-trained language models", type=int, default=768)
            parser.add_argument("-dp", "--dropout_p", dest="dropout_p", help="dropout probability", type=float, default=0.2)
            parser.add_argument("-hn", "--head_num", dest="head_num", help="number of multi-heads", type=int, default=12)

            parser.add_argument("-st","--step", dest="step", help="save/evaluate the model every step", type=int, default=10000)
            parser.add_argument("-ck","--checkpoint", dest="checkpoint", help="load the model from checkpoint before training/evaluating", type=int, default=0)
            parser.add_argument("-hst","--hold_step", dest="hold_step", help="keep training until step > hold_step", type=int, default=50000)
            parser.add_argument("-se", "--save_epoch", help="save after each epoch if declared", action="store_true", default=False)
            parser.add_argument("-lr", dest="lr", help="learning rate of non-bert modules", type=float, default=1e-4)
            parser.add_argument("-blr", "--bert_lr", dest="bert_lr", help="learning rate of bert based modules", type=float, default=6e-6)
            parser.add_argument("-vb", "--verbose", help="tailing name for tesrec", type=str, default="norm")

            parser.add_argument("-div", "--diversify", dest="diversify", help="whether to diversify selection with news representation", action="store_true", default=False)
            parser.add_argument("--descend_history", dest="descend_history", help="whether to order history by time in descending", action="store_true", default=False)
            parser.add_argument("--save_pos", dest="save_pos", help="whether to save token positions", action="store_true", default=False)
            parser.add_argument("--sep_his", dest="sep_his", help="whether to separate personalized terms from different news with an extra token", action="store_true", default=False)
            parser.add_argument("--full_attn", dest="full_attn", help="whether to interact among personalized terms (only in one-pass bert models)", action="store_true", default=False)
            parser.add_argument("--debias", dest="debias", help="whether to add a learnable bias to each candidate news's score", type=bool, default=True)
            parser.add_argument("--no_dedup", dest="no_dedup", help="whether to deduplicate tokens", action="store_true", default=False)
            parser.add_argument("--no_rm_punc", dest="no_rm_punc", help="whether to mask punctuations when selecting", action="store_true", default=False)
            parser.add_argument("--segment_embed", dest="segment_embed", help="whether to add an extra embedding to ps terms from the same historical news", action="store_true", default=False)

            parser.add_argument("-sm", "--smoothing", dest="smoothing", help="smoothing factor of tqdm", type=float, default=0.3)
            parser.add_argument("--num_workers", dest="num_workers", help="worker number of a dataloader", type=int, default=0)
            parser.add_argument("--shuffle", dest="shuffle", help="whether to shuffle the indices", action="store_true", default=False)
            parser.add_argument("--shuffle_pos", dest="shuffle_pos", help="whether to shuffle the candidate news and its negtive samples", action="store_true", default=False)
            parser.add_argument("--pin_memory", dest="pin_memory", help="whether to pin memory to speed up tensor transfer", action="store_true", default=False)
            parser.add_argument("--anomaly", dest="anomaly", help="whether to detect abnormal parameters", action="store_true", default=False)
            parser.add_argument("--scheduler", dest="scheduler", help="choose schedule scheme for optimizer", choices=["linear","none"], default="none")
            parser.add_argument("--warmup", dest="warmup", help="warmup steps of scheduler", type=int, default=10000)
            parser.add_argument("--interval", dest="interval", help="the step interval to update processing bar", default=10, type=int)
            parser.add_argument("--no_email", dest="no_email", help="whether to email the result", action='store_true', default=False)

            parser.add_argument("--npratio", dest="npratio", help="the number of unclicked news to sample when training", type=int, default=4)
            parser.add_argument("--metrics", dest="metrics", help="metrics for evaluating the model", type=str, default="")

            parser.add_argument("-emb", "--embedding", dest="embedding", help="choose embedding", choices=["bert","random","deberta"], default="bert")
            parser.add_argument("-encn", "--encoderN", dest="encoderN", help="choose news encoder", choices=["cnn","rnn","npa","fim","mha","bert"], default="cnn")
            parser.add_argument("-encu", "--encoderU", dest="encoderU", help="choose user encoder", choices=["avg","attn","cnn","lstm","gru","lstur","mha"], default="lstm")
            parser.add_argument("-slc", "--selector", dest="selector", help="choose history selector", choices=["recent","sfi"], default="sfi")
            parser.add_argument("-red", "--reducer", dest="reducer", help="choose document reducer", choices=["bm25","personalized","global","bow","entity","first","none","keyword"], default="personalized")
            parser.add_argument("-pl", "--pooler", dest="pooler", help="choose bert pooler", choices=["avg","attn","cls"], default="attn")
            parser.add_argument("-rk", "--ranker", dest="ranker", help="choose ranker", choices=["onepass","original","cnn","knrm"], default="original")
            parser.add_argument("-agg", "--aggregator", dest="aggregator", help="choose history aggregator, only used in TESRec", choices=["avg","attn","cnn","rnn","lstur","mha"], default=None)

            parser.add_argument("-k", dest="k", help="the number of the terms to extract from each news article", type=int, default=3)
            parser.add_argument("-b", "--bert", dest="bert", help="choose bert model", choices=["bert", "deberta", "unilm", "longformer", "bigbird", "reformer", "funnel", "synthesizer", "distill", "newsbert"], default="bert")

            parser.add_argument("-sd","--seed", dest="seed", default=42, type=int)
            parser.add_argument("-ws", "--world_size", dest="world_size", help="total number of gpus", default=0, type=int)

            args = parser.parse_args()

            args.cdd_size = args.npratio + 1
            args.metrics = "auc,mean_mrr,ndcg@5,ndcg@10".split(",") + [i for i in args.metrics.split(",") if i]

            # -1 means cpu
            if args.device == -1:
                args.device = "cpu"
                args.pin_memory = False

            if args.bert == 'unilm':
                args.unilm_path = args.path + 'bert_cache/UniLM/unilm2-base-uncased.bin'
                args.unilm_config_path = args.path + 'bert_cache/UniLM/unilm2-base-uncased-config.json'

            if args.recall_type is not None:
                # set to recall mode
                args.mode = "recall"

            if args.inspect_type is not None:
                args.mode = "inspect"

            # 1 for debug, 0 for evaluating only at the end of an epoch
            if args.step in [0,1]:
                args.hold_step = 0

            if args.scale == 'demo':
                args.no_email = True

        else:
            args = config

        if args.seed is not None:
            seed = args.seed
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

        for k,v in vars(args).items():
            if not k.startswith("__"):
                setattr(self, k, v)


    def __str__(self):
        return str({k:v for k,v in vars(self).items() if k not in hparam_list})


    def setup(self, rank):
        """
        set up distributed training and fix seeds
        """
        if self.world_size > 1:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            # prevent warning of transformers
            os.environ["TOKENIZERS_PARALLELISM"] = "True"
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

            # initialize the process group
            # set timeout to inf to prevent timeout error
            dist.init_process_group("nccl", rank=rank, world_size=self.world_size, timeout=timedelta(0, 1000000))

            # manager.rank will be invoked in creating DistributedSampler
            self.rank = rank
            # manager.device will be invoked in the model
            self.device = rank

        else:
            # one-gpu
            self.rank = -1

        if rank != "cpu":
            torch.cuda.set_device(rank)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)


    def prepare(self):
        """ prepare dataloader and several paths

        Returns:
            loaders(list of dataloaders): 0-loader_train/test/dev, 1-loader_dev
        """
        from .MIND import MIND, MIND_news, MIND_history
        from .utils import Partition_Sampler

        if self.rank in [-1, 0]:
            logger.info("Hyper Parameters are {}".format(self))
            logger.info("preparing dataset...")

        shuffle = self.shuffle
        pin_memory = self.pin_memory
        num_workers = self.num_workers

        if self.mode == "train":
            file_directory_train = self.path + "MIND/MIND{}_train/".format(self.scale)
            file_directory_dev = self.path + "MIND/MIND{}_dev/".format(self.scale)

            # construct whole dataset if needed
            if self.scale == 'whole' and not os.path.exists(file_directory_train):
                self.construct_whole_dataset()

            dataset_train = MIND(self, file_directory_train)
            dataset_dev = MIND(self, file_directory_dev)

            if self.world_size > 0:
                sampler_train = DistributedSampler(dataset_train, num_replicas=self.world_size, rank=self.rank, shuffle=True)
                sampler_dev = Partition_Sampler(dataset_dev, num_replicas=self.world_size, rank=self.rank)
            else:
                sampler_train = None
                sampler_dev = None

            loader_train = DataLoader(dataset_train, batch_size=self.batch_size, pin_memory=pin_memory,
                                    num_workers=num_workers, drop_last=False, shuffle=shuffle, sampler=sampler_train)
            loader_dev = DataLoader(dataset_dev, batch_size=1, pin_memory=pin_memory,
                                    num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler_dev)

            if self.fast:
                dataset_news = MIND_news(self, file_directory_dev)
                loader_news = DataLoader(dataset_news, batch_size=self.batch_size_news, pin_memory=pin_memory,
                        num_workers=num_workers, drop_last=False)

                return loader_train, loader_dev, loader_news

            else:
                loader_dev = DataLoader(dataset_dev, batch_size=1, pin_memory=pin_memory,
                                        num_workers=num_workers, drop_last=False, sampler=sampler_dev)
                return loader_train, loader_dev

        elif self.mode == "dev":
            file_directory_dev = self.path + "MIND/MIND{}_dev/".format(self.scale)
            dataset_dev = MIND(self, file_directory_dev)

            if self.world_size > 0:
                sampler_dev = Partition_Sampler(dataset_dev, num_replicas=self.world_size, rank=self.rank)
            else:
                sampler_dev = None

            loader_dev = DataLoader(dataset_dev, batch_size=1, pin_memory=pin_memory,
                        num_workers=num_workers, drop_last=False, sampler=sampler_dev)

            if self.fast:
                dataset_news = MIND_news(self, file_directory_dev)
                loader_news = DataLoader(dataset_news, batch_size=self.batch_size_news, pin_memory=pin_memory,
                        num_workers=num_workers, drop_last=False)

                return loader_dev, loader_news

            return loader_dev,

        elif self.mode == "inspect":
            # if case study
            if self.case:
                file_directory = "data/case/MINDlarge_{}/".format(self.inspect_type)
                dataset = MIND_history(self, file_directory)
                loader = DataLoader(dataset, batch_size=1, pin_memory=pin_memory,
                    num_workers=num_workers, drop_last=False)
            else:
                if self.inspect_type == "all":
                    file_directory = self.path + "MIND/MINDlarge_dev/"
                else:
                    file_directory = self.path + "MIND/MINDlarge_{}/".format(self.inspect_type)
                dataset = MIND_history(self, file_directory)
                loader = DataLoader(dataset, batch_size=1, pin_memory=pin_memory,
                    num_workers=num_workers, drop_last=False)

            return loader,

        elif self.mode == "test":
            file_directory_test = self.path + "MIND/MINDlarge_test/"
            dataset_test = MIND(self, file_directory_test)

            if self.world_size > 0:
                sampler_test = Partition_Sampler(dataset_test, num_replicas=self.world_size, rank=self.rank)
            else:
                sampler_test = None

            loader_test = DataLoader(dataset_test, batch_size=1, pin_memory=pin_memory,
                        num_workers=num_workers, drop_last=False, sampler=sampler_test)

            if self.fast:
                dataset_news = MIND_news(self, file_directory_test)
                loader_news = DataLoader(dataset_news, batch_size=self.batch_size_news, pin_memory=pin_memory,
                        num_workers=num_workers, drop_last=False)

                return loader_test, loader_news

            return loader_test,

        elif self.mode in ["encode", "analyse"]:
            file_directory = self.path + "MIND/MIND{}_dev/".format(self.scale)

            dataset = MIND_history(self, file_directory)
            sampler = None

            loader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=pin_memory,
                                    num_workers=num_workers, drop_last=False, shuffle=shuffle, sampler=sampler)

            return loader,

        elif self.mode == "recall":
            from .MIND import MIND_recall
            from .utils import collate_recall
            file_directory = self.path + "MIND/MINDlarge_dev/"

            dataset = MIND_recall(self, file_directory)
            loader = DataLoader(dataset, batch_size=self.batch_size_news, pin_memory=pin_memory,
                                    num_workers=num_workers, drop_last=False, collate_fn=collate_recall)

            return loader,


    def save(self, model, step, optimizer=None, best=False):
        """
            shortcut for saving the model and optimizer
        """
        if best:
            save_path = "data/model_params/{}/best.model".format(self.name)
        else:
            save_path = "data/model_params/{}/{}_step{}.model".format(self.name, self.scale, step)

        logger.info("saving model at {}...".format(save_path))
        model_dict = model.state_dict()

        save_dict = {}
        save_dict["model"] = model_dict
        save_dict["optimizer"] = optimizer.state_dict()

        torch.save(save_dict, save_path)


    def load(self, model, checkpoint, optimizer=None, strict=True):
        """
            shortcut for loading model and optimizer parameters
        """
        if checkpoint == 0:
            save_path = "data/model_params/{}/best.model".format(self.name)
            if not os.path.exists(save_path):
                logger.warning("not loading any checkpoints!")
                return
        else:
            save_path = "data/model_params/{}/{}_step{}.model".format(self.name, self.scale, checkpoint)

        logger.info("loading model from {}...".format(save_path))

        state_dict = torch.load(save_path, map_location=torch.device(model.device))

        if self.world_size <= 1:
            # in case we load a DDP model checkpoint to a non-DDP model
            model_dict = OrderedDict()
            pattern = re.compile("module.")

            for k,v in state_dict["model"].items():
                if re.search("module", k):
                    model_dict[re.sub(pattern, "", k)] = v
                else:
                    model_dict = state_dict["model"]
        else:
            model_dict = state_dict["model"]
            if not re.search("module", list(model_dict.keys())[0]):
                logger.warning("Loading a non-distributed model to a distributed one!")
                model = model.module

        model.load_state_dict(model_dict, strict=strict)

        if optimizer:
            optimizer.load_state_dict(state_dict["optimizer"])


    def _log(self, res):
        """
            wrap logging, skip logging results on MINDdemo
        """

        # logger.info("evaluation results:{}".format(res))
        with open("performance.log", "a+") as f:
            d = {"name": self.name}
            for k, v in vars(self).items():
                if k not in hparam_list:
                    d[k] = v

            name = str(d) + "\n"
            content = str(res) + "\n\n"
            f.write(name)
            f.write(content)

            if not self.no_email:
                try:
                    from data.configs.email import email,password

                    subject = "[Performance Report] {} : {}".format(d["name"], res["auc"])
                    email_server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
                    email_server.login(email, password)
                    message = "Subject: {}\n\n{}".format(subject,name + content)
                    email_server.sendmail(email, email, message)
                    email_server.close()
                except:
                    logger.info("error in connecting SMTP")


    def _get_loss(self, model):
        """
            get loss function for model
        """
        if model.training:
            loss = nn.NLLLoss()
        else:
            loss = nn.BCELoss()

        return loss


    def _get_optim(self, model, loader_train_length):
        """
            get optimizer and scheduler
        """
        if self.world_size > 1:
            model = model.module

        base_params = []
        bert_params = []
        for name, param in model.named_parameters():
            if re.search("bert", name):
                bert_params.append(param)
            else:
                base_params.append(param)

        optimizer = optim.Adam([
            {
                "params": base_params,
                "lr": self.lr #lr_schedule(args.lr, 1, args)
            },
            {
                "params": bert_params,
                "lr": self.bert_lr #lr_schedule(args.pretrain_lr, 1, args)
            }
        ])

        scheduler = None
        if self.scheduler == "linear":
            total_steps = loader_train_length * self.epochs
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = self.warmup,
                                            num_training_steps = total_steps)

        return optimizer, scheduler


    @torch.no_grad()
    def _eval(self, model, loader_dev):
        """ making prediction and gather results into groups according to impression_id

        Args:
            loader_dev(torch.utils.data.DataLoader): provide data

        Returns:
            impr_indexes: impression ids
            labels: labels
            preds: preds
        """
        if self.rank in [-1, 0]:
            logger.info("evaluating...")

        impr_indexes = []
        labels = []
        preds = []

        for x in tqdm(loader_dev, smoothing=self.smoothing, ncols=120, leave=True):
            impr_indexes.extend(x["impr_index"].tolist())
            preds.extend(model(x)[0].tolist())
            labels.extend(x["label"].tolist())

        # collect result across gpus when distributed evaluating
        if self.world_size > 1:
            dist.barrier()
            outputs = [None for i in range(self.world_size)]
            dist.all_gather_object(outputs, (impr_indexes, labels, preds))

            if self.rank == 0:
                impr_indexes = []
                labels = []
                preds = []
                for output in outputs:
                    impr_indexes.extend(output[0])
                    labels.extend(output[1])
                    preds.extend(output[2])

                labels, preds = _group_lists(impr_indexes, labels, preds)

        else:
            labels, preds = _group_lists(impr_indexes, labels, preds)


        return labels, preds


    @torch.no_grad()
    def _eval_fast(self, model, loaders):
        """
        1. encode and save news
        2. compute scores by look-up tables and dot product

        Args:
            model
            loaders
                0: loader_test
                1: loader_news
        """
        # if the model is an instance of DDP, then we have to access its module attribute
        if self.world_size > 1:
            model = model.module

        # encode and save news representations only on the master node
        if self.rank in [-1, 0]:
            cache_directory = "data/cache/tensors/{}/{}/dev/".format(self.name, self.scale)
            os.makedirs(cache_directory, exist_ok=True)
            logger.info("fast evaluate, encoding news...")

            model.init_encoding()
            news_reprs = torch.zeros(self.get_news_num() + 1, model.hidden_dim, device=model.device)

            for x in tqdm(loaders[1], smoothing=self.smoothing, ncols=120, leave=True):
                news_repr = model.encode_news(x).squeeze(-2)
                news_reprs[x['cdd_id']] = news_repr

            # must save for other processes to load
            torch.save(news_reprs, cache_directory + "news.pt")
            del news_reprs
            model.destroy_encoding()

        if self.world_size > 1:
            dist.barrier()

        model.init_embedding()
        impr_indexes = []
        labels = []
        preds = []
        for x in tqdm(loaders[0], smoothing=self.smoothing, ncols=120, leave=True):
            impr_indexes.extend(x["impr_index"].tolist())
            preds.extend(model.predict_fast(x).tolist())
            labels.extend(x["label"].tolist())

        model.destroy_embedding()

        # collect result across gpus when distributed evaluating
        if self.world_size > 1:
            dist.barrier()
            outputs = [None for i in range(self.world_size)]
            dist.all_gather_object(outputs, (impr_indexes, labels, preds))

            if self.rank == 0:
                impr_indexes = []
                labels = []
                preds = []
                for output in outputs:
                    impr_indexes.extend(output[0])
                    labels.extend(output[1])
                    preds.extend(output[2])

                labels, preds = _group_lists(impr_indexes, labels, preds)

        else:
            labels, preds = _group_lists(impr_indexes, labels, preds)

        return labels, preds


    def evaluate(self, model, loaders, load=False, log=True, optimizer=None, scheduler=None):
        """Evaluate the given file and returns some evaluation metrics.

        Args:
            self(dict)
            model
            loaders(torch.utils.data.DataLoader):
                0: loader_dev
                1: loader_news
            loading(bool): whether to load model
            log(bool): whether to log

        Returns:
            res(dict): A dictionary contains evaluation metrics.
        """
        model.eval()

        # load saved model
        if load:
            self.load(model, self.checkpoint, optimizer)

        # protect non-master node to return an object
        res = None

        if self.fast:
            # fast evaluate
            labels, preds = self._eval_fast(model, loaders)
        else:
            # slow evaluate
            labels, preds = self._eval(model, loaders[0])

        # compute metrics only on the master node
        if self.rank in [0, -1]:
            res = cal_metric(labels, preds, self.metrics)

            logger.info("\nevaluation result of {} is {}".format(self.name, res))

            if log and self.rank in [0, -1]:
                res["step"] = self.checkpoint
                self._log(res)

        return res


    def _train(self, model, loaders, optimizer, loss_func, scheduler=None):
        """ train model and evaluate on validation set once every save_step
        Args:
            dataloader(torch.utils.data.DataLoader): provide data
            optimizer(list of torch.nn.optim): optimizer for training
            loss_func(torch.nn.Loss): loss function for training
            schedulers
            writer(torch.utils.tensorboard.SummaryWriter): tensorboard writer
            interval(int): within each epoch, the interval of training steps to display loss
            save_step(int): how many steps to save the model
        Returns:
            model: trained model
        """
        steps = 0
        interval = self.interval
        save_epoch = self.save_epoch

        if self.scale == "demo":
            save_step = len(loaders[0]) - 1
            self.fast = False
            # save_step = 1
        else:
            if self.step == 0:
                save_step = len(loaders[0]) - 1
            else:
                save_step = self.step

        print(save_step)

        distributed = self.world_size > 1
        # if self.tb:
        #     writer = SummaryWriter("data/tb/{}/{}/{}/".format(
        #         self.name, self.scale, datetime.now().strftime("%Y%m%d-%H")))

        best_res = {"auc":0}

        if self.rank in [0, -1]:
            logger.info("training {}...".format(self.name))
            # logger.info("total training step: {}".format(self.epochs * len(loaders[0])))

        for epoch in range(self.epochs):
            epoch_loss = 0
            if distributed:
                loaders[0].sampler.set_epoch(epoch)
            tqdm_ = tqdm(loaders[0], smoothing=self.smoothing, ncols=120, leave=True)

            for step, x in enumerate(tqdm_):

                optimizer.zero_grad(set_to_none=True)

                pred = model(x)[0]
                label = x["label"].to(model.device)

                loss = loss_func(pred, label)
                epoch_loss += float(loss)

                loss.backward()

                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                optimizer.step()

                if scheduler:
                    scheduler.step()

                if step % interval == 0 and step > 0:
                    tqdm_.set_description(
                        "epoch: {:d},  step: {:d},  total_step: {:d},  loss: {:.4f}".format(epoch + 1, step, steps, epoch_loss / step))
                    # if writer:
                    #     for name, param in model.named_parameters():
                    #         writer.add_histogram(name, param, step)

                    #     writer.add_scalar("data_loss",
                    #                     total_loss/total_steps)

                if steps % save_step == 0 and (steps in [140000,150000,180000,200000,210000,220000,230000,240000] and self.scale == "whole" or steps > self.hold_step and self.scale == "large" or steps > 0 and self.scale == "demo"):
                    print("\n")
                    with torch.no_grad():
                        result = self.evaluate(model, loaders[1:], log=False)

                        if self.rank in [0,-1]:
                            if save_epoch:
                                self.save(model, steps, optimizer)

                            result["step"] = steps

                            if result["auc"] > best_res["auc"]:
                                best_res = result
                                self.save(model, steps, optimizer, best=True)
                                self._log(result)

                        # prevent SIGABRT
                        if distributed:
                            dist.barrier()
                    # continue training
                    model.train()

                steps += 1

        return best_res


    def train(self, model, loaders):
        """ train and evaluate sequentially

        Args:
            model(torch.nn.Module): the model to be trained
            loaders(list): list of torch.utils.data.DataLoader
            en: shell parameter
        """

        model.train()

        if self.rank in [-1, 0]:
            # in case the folder does not exists, create one
            os.makedirs("data/model_params/{}".format(self.name), exist_ok=True)

        loss_func = self._get_loss(model)
        optimizer, scheduler = self._get_optim(model, len(loaders[0]))

        if self.checkpoint:
            self.load(model, self.checkpoint, optimizer)

        if self.anomaly:
            with torch.autograd.set_detect_anomaly(True):
                res = self._train(model, loaders, optimizer, loss_func, scheduler=scheduler)
        else:
            res = self._train(model, loaders, optimizer, loss_func, scheduler=scheduler)

        if self.rank in [-1,0]:
            logger.info("Best result: {}".format(res))
            self._log(res)


    @torch.no_grad()
    def _test(self, model, loader_test):
        """
        regular test without pre-encoding
        """
        impr_indexes = []
        preds = []

        for x in tqdm(loader_test, smoothing=self.smoothing, ncols=120, leave=True):
            impr_indexes.extend(x["impr_index"].tolist())
            preds.extend(model(x)[0].tolist())

        if self.world_size > 1:
            dist.barrier()
            outputs = [None for i in range(self.world_size)]
            dist.all_gather_object(outputs, (impr_indexes, preds))

            if self.rank == 0:
                impr_indexes = []
                preds = []
                for output in outputs:
                    impr_indexes.extend(output[0])
                    preds.extend(output[1])

                preds = _group_lists(impr_indexes, preds)[0]
        else:
            preds = _group_lists(impr_indexes, preds)[0]

        return preds


    @torch.no_grad()
    def _test_fast(self, model, loaders):
        """
        1. encode and save news
        2. compute scores by look-up tables and dot product

        Args:
            model
            loaders
                0: loader_test
                1: loader_news
        """
        if self.world_size > 1:
            model = model.module

        if self.rank in [-1, 0]:
            cache_directory = "data/cache/tensors/{}/{}/test/".format(self.name, self.scale)
            os.makedirs(cache_directory, exist_ok=True)
            logger.info("encoding news...")

            model.init_encoding()
            news_reprs = torch.zeros(self.get_news_num() + 1, model.hidden_dim, device=model.device)

            for x in tqdm(loaders[1], smoothing=self.smoothing, ncols=120, leave=True):
                news_repr = model.encode_news(x).squeeze(-2)
                news_reprs[x['cdd_id']] = news_repr

            torch.save(news_reprs, cache_directory + "news.pt")
            del news_reprs
            model.destroy_encoding()
            logger.info("inferring...")

        if self.world_size > 1:
            dist.barrier()

        model.init_embedding()
        impr_indexes = []
        preds = []
        for x in tqdm(loaders[0], smoothing=self.smoothing, ncols=120, leave=True):
            impr_indexes.extend(x["impr_index"].tolist())
            preds.extend(model.predict_fast(x).tolist())

        model.destroy_embedding()

        if self.world_size > 1:
            dist.barrier()
            outputs = [None for i in range(self.world_size)]
            dist.all_gather_object(outputs, (impr_indexes, preds))

            if self.rank == 0:
                impr_indexes = []
                preds = []
                for output in outputs:
                    impr_indexes.extend(output[0])
                    preds.extend(output[1])

                preds = _group_lists(impr_indexes, preds)[0]
        else:
            preds = _group_lists(impr_indexes, preds)[0]

        return preds


    def test(self, model, loaders):
        """ test the model on test dataset of MINDlarge

        Args:
            model
            loader_test: DataLoader of MINDlarge_test
        """
        model.eval()

        self.load(model, self.checkpoint)

        if self.rank in [0,-1]:
            logger.info("testing...")

        if self.fast:
            # fast evaluate
            preds = self._test_fast(model, loaders)
        else:
            # slow evaluate
            preds = self._test(model, loaders[0])

        if self.rank in [0, -1]:
            save_directory = "data/results/{}".format(self.name + "/{}_step{}".format(self.scale, self.checkpoint))
            os.makedirs(save_directory, exist_ok=True)

            save_path = save_directory + "/prediction.txt"

            index = 1
            with open(save_path, "w") as f:
                for pred in preds:
                    array = np.asarray(pred)
                    rank_list = ss.rankdata(1 - array, method="ordinal")
                    line = str(index) + " [" + ",".join([str(i)
                                                    for i in rank_list]) + "]" + "\n"
                    f.write(line)
                    index += 1

            logger.info("written to prediction at {}!".format(save_path))


    @torch.no_grad()
    def inspect(self, model, loaders):
        """
        inspect personalized terms
        """
        from transformers import AutoTokenizer

        model.eval()
        logger.info("inspecting {}...".format(self.name))

        loader_inspect = loaders[0]

        self.load(model, self.checkpoint)

        with open(loader_inspect.dataset.news_cache_directory + "news.pkl", "rb") as f:
            news = pickle.load(f)["encoded_news"][:, :self.signal_length]
        with open(loader_inspect.dataset.news_cache_directory + "bm25.pkl", "rb") as f:
            bm25_terms = pickle.load(f)["encoded_news"][:, :10]
        with open(loader_inspect.dataset.news_cache_directory + "entity.pkl", "rb") as f:
            entities = pickle.load(f)["encoded_news"][:, :10]

        t = AutoTokenizer.from_pretrained(self.get_bert_for_load(), cache_dir=self.path + "bert_cache/")

        logger.info("press <ENTER> to continue")

        if self.news is not None:
            if self.news.lower().startswith("n"):
                target_news = loader_inspect.dataset.nid2index[self.news.upper()]
            else:
                target_news = int(self.news)

        if self.inspect_type == "all":
            keyterm_matrix = np.zeros((self.get_news_num(), 10), dtype=int)
            user_matrix = np.zeros(self.get_news_num(), dtype=int)

            for x in loader_inspect:
                term_indexes = model.encode_user(x)[1].cpu().numpy()
                # hs, 1
                news_ids = x['his_id'][0].squeeze(-1).numpy()
                user_ids = x['user_id'].numpy()
                term_indexes = term_indexes[0]

                for his_id, term_index, user_id in zip(news_ids, term_indexes, user_ids):
                    reference_term_index = keyterm_matrix[his_id]
                    if np.sum(reference_term_index != term_index) > 4 and reference_term_index.sum() > 0:
                        # strip [CLS]
                        his_news = news[his_id][1:]
                        prev_term = his_news[reference_term_index]
                        now_term = his_news[term_index]
                        prev_user = user_matrix[his_id]
                        print("*************************{}:{}->{}******************************".format(his_id, prev_user, user_id))
                        print(t.decode(prev_term, skip_special_tokens=True))
                        print(t.decode(now_term, skip_special_tokens=True))
                        print("[original news]\n\t {}".format(t.decode(news[his_id][:self.signal_length], skip_special_tokens=True)))
                        print(user_id)
                        input()
                        break

                    else:
                        keyterm_matrix[his_id] = term_index
                        user_matrix[his_id] = user_id

        else:
            for x in loader_inspect:
                term_indexes = model.encode_user(x)[1]

                his_encoded_index = x["his_encoded_index"][0][:, 1:]
                if self.reducer == "bow":
                    his_encoded_index = his_encoded_index[ :, 1:, 0]
                term_indexes = term_indexes[0].cpu().numpy()
                his_ids = x["his_id"][0].tolist()
                user_index = x["user_id"][0]

                if self.news is not None:
                    for j, his_id in enumerate(his_ids):
                        # skip untargetted news
                        if his_id == target_news:
                            break

                    his_encoded_index = [his_encoded_index[j]]
                    term_indexes = [term_indexes[j]]
                    his_ids = [his_id]

                for his_token_ids, term_index, his_id in zip(his_encoded_index, term_indexes, his_ids):

                    print("*************************{}******************************".format(user_index.item()))
                    tokens = t.convert_ids_to_tokens(his_token_ids)

                    terms = tokens
                    terms = np.asarray(terms)

                    ps_terms = terms[term_index]

                    if his_id == 0:
                        break
                    else:
                        print("[personalized terms]\n\t {}".format(" ".join(ps_terms)))
                        print("[bm25 terms]\n\t {}".format(t.decode(bm25_terms[his_id], skip_special_tokens=True)))
                        print("[entities]\n\t {}".format(t.decode(entities[his_id], skip_special_tokens=True)))
                        print("[original news]\n\t {}".format(t.decode(news[his_id][:self.signal_length], skip_special_tokens=True)))

                    command = input()
                    if command == "n":
                        break
                    elif command == "q":
                        return


    @torch.no_grad()
    def encode(self, model, loaders):
        """
        encode user
        """
        model.eval()
        logger.info("encoding users...")

        self.load(model, self.checkpoint)

        # user_reprs = []
        start_time = time.time()
        for x in tqdm(loaders[0], smoothing=self.smoothing, ncols=120, leave=True):
            user_repr = model.encode_user(x)[0].squeeze(-2)

        end_time = time.time()
        logger.info("total encoding time: {}".format(end_time - start_time))


    @torch.no_grad()
    def recall(self, model, loaders):
        """
        recall from the given corpus
        """

        device = torch.device(model.device)

        if self.recall_type == "d":
            logger.info("dense recalling by user representation...")
            import faiss
            self.load(model, self.checkpoint)

            news_set = torch.load('data/recall/news.pt', map_location=device)
            print(len(news_set))
            try:
                news_reprs = torch.load("data/cache/tensors/{}/large/dev/news.pt".format(self.name), map_location=device)
            except:
                raise FileNotFoundError("please run 'python xxx.py -m dev -ck=xx' first!")

            news = news_reprs[news_set]

            del news_set
            del news_reprs
            ngpus = faiss.get_num_gpus()

            INDEX = faiss.IndexFlatIP(news.size(-1))
            # INDEX = faiss.index_cpu_to_all_gpus(INDEX)
            INDEX = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.device, INDEX)
            INDEX.add(news.cpu().numpy())

            recalls = np.array([0.,0.,0.])
            count = 0

            model.init_embedding()
            for x in tqdm(loaders[0], smoothing=self.smoothing, ncols=120, leave=True):
                # t1 = time.time()
                user_repr = model.encode_user(x)[0].squeeze(-2).cpu().numpy()
                # t2 = time.time()
                _, index = INDEX.search(user_repr, 100)
                # t3 = time.time()

                for i,cdd_id in enumerate(x['cdd_id']):
                    recalls[0] += np.sum(np.in1d(cdd_id, index[i][:10]))
                    recalls[1] += np.sum(np.in1d(cdd_id, index[i][:50]))
                    recalls[2] += np.sum(np.in1d(cdd_id, index[i][:100]))
                    count += len(cdd_id)
                t4 = time.time()
                # print("transfer time:{}, search time:{}, operation time:{}".format(t2-t1, t3-t2, t4-t3))
            model.destroy_embedding()


        elif self.recall_type == "s":
            logger.info("sparse recalling by extracted terms...")
            try:
                inverted_index = torch.load("data/recall/inverted_index_bm25.pt", map_location=device)
            except:
                from .utils import BM25_token, construct_inverted_index
                with open("data/cache/MIND/news/bert/MINDlarge_dev/news.pkl", "rb") as f:
                    news = pickle.load(f)['encoded_news']
                news_set = torch.load('data/recall/news.pt', map_location="cpu").numpy()
                news_collection = news[news_set]
                # initialize inverted index
                bm25 = BM25_token(news_collection)
                inverted_index = construct_inverted_index(news_collection, bm25).to(device)

            self.load(model, self.checkpoint)

            recalls = torch.zeros(3, device=device)
            count = 0
            # total news number
            news_num = 6997
            for x in tqdm(loaders[0], smoothing=self.smoothing, ncols=120, leave=True):
                t1 = time.time()
                kid = model.encode_user(x)[1]
                batch_size = x['his_encoded_index'].size(0)
                ps_terms = x['his_encoded_index'][:, :, 1:].to(self.device).gather(index=kid, dim=-1).view(batch_size, -1).cpu().numpy()
                t2 = time.time()

                for query, cdd_id in zip(ps_terms, x['cdd_id']):
                    # Q, N, 2
                    recalled_news_ids_and_scores = inverted_index[query]

                    news_score_all = torch.zeros(news_num + 1, device=model.device)
                    recalled_news_ids = recalled_news_ids_and_scores[:, :, 0].long()
                    recalled_news_scores = recalled_news_ids_and_scores[:, :, 1]
                    for i,j in zip(recalled_news_ids, recalled_news_scores):
                        news_score_all[i] += j

                    recalled_news_ids = news_score_all.argsort(descending=True)
                    # recalls += torch.sum(torch.from_numpy(cdd_id).to(device).unsqueeze(-1) == recalled_news_ids)

                    recalls[0] += torch.sum(torch.from_numpy(cdd_id).to(device).unsqueeze(-1) == recalled_news_ids[:10])
                    recalls[1] += torch.sum(torch.from_numpy(cdd_id).to(device).unsqueeze(-1) == recalled_news_ids[:50])
                    recalls[2] += torch.sum(torch.from_numpy(cdd_id).to(device).unsqueeze(-1) == recalled_news_ids[:100])

                    count += len(cdd_id)

                t3 = time.time()
                # print("transfer time:{}, search time:{}".format(t2-t1, t3-t2))


        elif self.recall_type == "sd":
            logger.info("sparse recalling then dense ranking by extracted terms and user representation respectively...")

            news_set = torch.load('data/recall/news.pt', map_location=device)
            news_reprs = torch.load("data/cache/tensors/{}/large/dev/news.pt".format(self.name), map_location=device)
            news = news_reprs[news_set]
            news = torch.cat([news, torch.zeros(1, news.size(-1), device=device)], dim=0)

            try:
                inverted_index = torch.load("data/recall/inverted_index_bm25.pt", map_location=device)
            except:
                from .utils import BM25_token, construct_inverted_index
                with open("data/cache/MIND/news/bert/MINDlarge_dev/news.pkl", "rb") as f:
                    news_ids = pickle.load(f)['encoded_news']
                news_collection = news_ids[news_set.cpu().numpy()]
                # initialize inverted index
                bm25 = BM25_token(news_collection)
                inverted_index = construct_inverted_index(news_collection, bm25).to(device)

            self.load(model, self.checkpoint)

            count = 0
            print(news.shape)
            recalls = torch.zeros(3, device=device)
            for x in tqdm(loaders[0], smoothing=self.smoothing, ncols=120, leave=True):
                t1 = time.time()
                user_reprs, kid = model.encode_user(x)
                batch_size = x['his_encoded_index'].size(0)
                ps_terms = x['his_encoded_index'][:, :, 1:].to(device).gather(index=kid, dim=-1).view(batch_size, -1)
                t2 = time.time()

                for query, user_repr, cdd_id in zip(ps_terms, user_reprs, x['cdd_id']):
                    recalled_news_ids = inverted_index[query][:, :, 0].long().unique()

                    # N, D
                    recalled_news = news[recalled_news_ids]
                    # N, 1
                    recalled_news_score = recalled_news.matmul(user_repr.transpose(-1,-2)).squeeze(-1)
                    # sort
                    recalled_news_ids = recalled_news_ids[recalled_news_score.argsort(descending=True)][:100]

                    recalls[0] += torch.sum(torch.from_numpy(cdd_id).to(device).unsqueeze(-1) == recalled_news_ids[:10])
                    recalls[1] += torch.sum(torch.from_numpy(cdd_id).to(device).unsqueeze(-1) == recalled_news_ids[:50])
                    recalls[2] += torch.sum(torch.from_numpy(cdd_id).to(device).unsqueeze(-1) == recalled_news_ids[:100])
                    count += len(cdd_id)

                t3 = time.time()

                # print("transfer time:{}, search time:{}".format(t2-t1, t3-t2))

        recall_rates = recalls / count
        logger.info("recall@10 : {}; recall@50 : {}; recall@100 : {}".format(recall_rates[0], recall_rates[1], recall_rates[2]))


    @torch.no_grad()
    def collect_kid(self, model, loaders):
        """
        1. collect end-to-end extracted terms' position
        2. compute recall rate of entity
        3. compute recall rate of non-entity
        """
        if self.rank in [-1, 0]:
            logger.info("collecting extracted terms' position distribution...")

        self.load(model, self.checkpoint)

        def construct_heuristic_extracted_term_indice():
            logger.info("constructing heuristicly extracted term indices...")
            news = pickle.load(open('data/cache/MIND/news/bert/MINDlarge_dev/news.pkl','rb'))['encoded_news']
            bm25 = pickle.load(open('data/cache/MIND/news/bert/MINDlarge_dev/bm25.pkl','rb'))['encoded_news']
            entity = pickle.load(open('data/cache/MIND/news/bert/MINDlarge_dev/entity.pkl','rb'))['encoded_news']

            bm25_indice = []
            entity_indice = []

            for n,b,e in zip(news, bm25, entity):
                # strip [CLS]
                bm25_index = []
                entity_index = []

                # for each token in the first 3 bm25, find its position in the original news
                for tok1 in b[1:self.k + 1]:
                    found = False
                    for i,tok2 in enumerate(n[1:100]):
                        if tok1 == tok2:
                            bm25_index.append(i)
                            found = True
                            break
                    if not found:
                        bm25_index.append(-1)

                for tok1 in e[1:self.k + 1]:
                    found = False
                    for i,tok2 in enumerate(n[1:100]):
                        if tok1 == tok2:
                            found = True
                            entity_index.append(i)
                            break
                    if not found:
                        entity_index.append(-1)

                bm25_indice.append(bm25_index)
                entity_indice.append(entity_index)

            bm25_indice = np.asarray(bm25_indice)
            entity_indice = np.asarray(entity_indice)

            os.makedirs("data/cache/kid", exist_ok=True)
            np.save("data/cache/kid/bm25.npy", bm25_indice)
            np.save("data/cache/kid/entity.npy", entity_indice)
            return bm25_indice, entity_indice

        try:
            entity_indice = np.load('data/cache/kid/entity.npy')
        except:
            _, entity_indice = construct_heuristic_extracted_term_indice()

        # bm25_positions = np.array([0] * (self.signal_length - 1))
        # entity_positions = np.array([0] * (self.signal_length - 1))
        device = model.device
        entity_indice = torch.from_numpy(entity_indice).to(device)
        kid_positions = torch.zeros(self.signal_length + 1, dtype=torch.int32, device=device)

        count = 0
        entity_recall = 0.

        for x in tqdm(loaders[0], smoothing=self.smoothing, ncols=120, leave=True):
            # strip off [CLS] and [SEP]
            kids = model.encode_user(x)[1]
            kids = kids.masked_fill(~(x['his_mask'].to(self.device).bool()), self.signal_length)
            for kid, his_id in zip(kids.reshape(-1, self.k), x['his_id'].to(device).view(-1)):
                kid_positions[kid] += 1

                # entity_index = entity_indice[his_id]
                # entity_recall += torch.sum(entity_index.unsqueeze(-1) == kid)
                # count += 1

        # logger.info("entity recall rate: {}".format(entity_recall / count))
        torch.save(kid_positions, "data/cache/kid/kid_pos.pt")

        # pos = []
        # for x in tqdm(loaders[0], smoothing=self.smoothing, ncols=120, leave=True):
        #     # strip off [CLS] and [SEP]
        #     kids = model.encode_user(x)[1]
        #     pos.append(kids.view(-1))
        # unique, count = torch.cat(pos, dim=-1).unique(return_counts=True)
        # result = dict(zip(unique.tolist(), count.tolist()))
        # json.dump(result, "data/cache/kid_pos.json")


    def get_user_num(self):
        user_num_map = {
            "demo": 2146,
            "small": 94057,
            "large": 876956,
            "whole": 876956
        }
        return user_num_map[self.scale]


    def get_news_num(self):
        """
        get the news number of validation/test set
        """
        news_num_map = {
            "demo":{
                "train": 42416,
                "dev": 42416,
                "inspect": 42416,
                "test": 120961
            },
            "small":{
                "train": 42416,
                "dev": 42416,
                "inspect": 42416,
                "test": 120961
            },
            "large":{
                "train": 72023,
                "dev": 72023,
                "inspect": 72023,
                "test": 120961
            },
            "whole":{
                "train": 72023,
                "dev": 72023,
                "inspect": 72023,
                "test": 120961
            }
        }
        return news_num_map[self.scale][self.mode]


    def get_news_file_for_load(self):
        """
        map different reducer to different news cache file
        """
        reducer_map = {
            "none": "news.pkl",
            "personalized": "news.pkl",
            "global": "news.pkl",
            "bm25": "bm25.pkl",
            "bow": "news.pkl",
            "entity": "entity.pkl",
            "first": "news.pkl",
            "keyword": "keyword.pkl"
        }
        return reducer_map[self.reducer]


    def get_bert_for_load(self):
        """
        transfer unilm to bert
        """
        bert_map = {
            "bert": "bert-base-uncased",
            "deberta": "microsoft/deberta-base",
            "unilm": "bert-base-uncased",
            "longformer": "allenai/longformer-base-4096",
            "bigbird": "google/bigbird-roberta-base",
            "reformer": "google/reformer-crime-and-punishment",
            "funnel": "funnel-transformer/small-base",
            "synthesizer": "bert-base-uncased",
            "distill": "distilbert-base-uncased",
            "newsbert": "bert-base-uncased"
        }
        return bert_map[self.bert]


    def get_bert_for_cache(self):
        """
        path to save cached tokenization file, transfer unilm to bert
        """
        bert_map = {
            "bert": "bert",
            "deberta": "deberta",
            "unilm": "bert",
            "longformer": "longformer",
            "bigbird": "bigbird",
            "reformer": "reformer",
            "funnel": "bert",
            "synthesizer": "bert",
            "distill": "bert",
            "newsbert": "bert"
        }
        return bert_map[self.bert]


    def get_special_token_id(self, token):
        special_token_map = {
            "bert":{
                "[PAD]": 0,
                "[CLS]": 101,
                "[SEP]": 102,
            },
            "deberta":{
                "[PAD]": 0,
                "[CLS]": 1,
                "[SEP]": 2,
            },
            "unilm":{
                "[PAD]": 0,
                "[CLS]": 101,
                "[SEP]": 102,
            },
            "longformer":{
                "[PAD]": 1,
                "[CLS]": 0,
                "[SEP]": 2,
            },
            "bigbird":{
                "[PAD]": 0,
                "[CLS]": 65,
                "[SEP]": 66
            },
            "reformer":{
                "[PAD]": 2,
                "[CLS]": 1,
                "[SEP]": 2
            },
            "funnel":{
                "[PAD]": 0,
                "[CLS]": 101,
                "[SEP]": 102,
            },
            "synthesizer":{
                "[PAD]": 0,
                "[CLS]": 101,
                "[SEP]": 102,
            },
            "distill":{
                "[PAD]": 0,
                "[CLS]": 101,
                "[SEP]": 102,
            },
            "newsbert":{
                "[PAD]": 0,
                "[CLS]": 101,
                "[SEP]": 102,
            },
        }
        return special_token_map[self.bert][token]


    def get_activation_func(self):
        act_map = {
            "bert": torch.tanh,
            "deberta": nn.functional.gelu,
            "unilm": torch.tanh
        }
        return act_map[self.bert]


    def get_max_length_for_truncating(self):
        """
        get max length in Truncating_Reducer
        """
        # subtract 1 because [CLS]
        length_map = {
            "bert": (512, 10),
            "deberta": (512, 10),
            "unilm": (512, 10),
            "longformer": (1024, 21),
            "bigbird": (1024, 21),
            "reformer": (1280, 26),
        }
        return length_map[self.bert]


    def construct_nid2idx(self, scale=None, mode=None):
        """
            Construct news to newsID dictionary, index starting from 1
            VERY IMPORTANT!!!
            The nid2idx dictionary must follow the original order of news in news.tsv
        """
        logger.info("mapping news id to news index...")
        nid2index = {}
        mind_path = self.path + "MIND"

        if scale is None:
            scale = self.scale
        if mode is None:
            mode = self.mode
        news_file = mind_path+"/MIND"+ scale+ "_" + mode + "/news.tsv"

        news_df = pd.read_table(news_file, index_col=None, names=[
                                "newsID", "category", "subcategory", "title", "abstract", "url", "entity_title", "entity_abstract"], quoting=3)

        for v in news_df["newsID"]:
            if v in nid2index:
                continue
            nid2index[v] = len(nid2index) + 1

        h = open("data/dictionaries/nid2idx_{}_{}.json".format(scale, mode), "w")
        json.dump(nid2index, h, ensure_ascii=False)
        h.close()


    def construct_uid2idx(self, scale=None):
        """
            Construct user to userID dictionary, index starting from 1
        """
        logger.info("mapping user id to user index...")
        uid2index = {}
        user_df_list = []
        mind_path = self.path + "MIND"
        if scale is None:
            scale = self.scale
        behavior_file_list = [mind_path+"/MIND" + scale + "_" + mode + "/behaviors.tsv" for mode in ["train","dev","test"]]

        if scale == "small":
            behavior_file_list = behavior_file_list[:2]

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


    def construct_cddidx_for_recall(self, imprs):
        """
        map original cdd id to the one that's limited to the faiss index
        """
        logger.info("mapping original cdd id to the one that's limited to the faiss index...")
        os.makedirs("data/recall", exist_ok=True)
        # get the index of news that appeared in impressions
        try:
            news_set = torch.load('data/recall/news.pt', map_location='cpu')
        except:
            news_set = set()

            for impr in imprs:
                news_set.update(impr[1])

            news_set = torch.tensor(list(news_set))
            torch.save(news_set, "data/recall/news.pt")

        news_set = news_set.tolist()
        cddid2idx = {}
        for i,x in enumerate(news_set):
            cddid2idx[x] = i

        with open("data/recall/cddid2idx_recall.pkl", "wb") as f:
            pickle.dump(cddid2idx, f)


    def construct_whole_dataset(self):
        """
        join training set and validation set
        """
        logger.info("merging MINDlarge_train and MINDlarge_dev into MINDwhole_train")

        whole_directory = self.path + "MIND/MINDwhole_train/"
        os.makedirs(whole_directory, exist_ok=True)
        whole_directory = self.path + "MIND/MINDwhole_dev/"
        os.makedirs(whole_directory, exist_ok=True)

        behav_path_train = self.path + "MIND/MINDlarge_train/behaviors.tsv"
        behav_path_dev = self.path + "MIND/MINDlarge_dev/behaviors.tsv"
        behav_path_whole_train = self.path + "MIND/MINDwhole_train/behaviors.tsv"
        behav_path_whole_dev = self.path + "MIND/MINDwhole_dev/behaviors.tsv"

        f = open(behav_path_whole_train, "w")
        h = open(behav_path_whole_dev, "w")
        with open(behav_path_train, "r") as g:
            for line in g:
                f.write(line)
        with open(behav_path_dev, "r") as g:
            count = 0
            for line in g:
                if count < 200000:
                    f.write(line)
                else:
                    h.write(line)

                count += 1

        f.close()
        h.close()

        news_path_train = self.path + "MIND/MINDlarge_train/news.tsv"
        news_path_dev = self.path + "MIND/MINDlarge_dev/news.tsv"
        news_path_whole_train = self.path + "MIND/MINDwhole_train/news.tsv"
        news_path_whole_dev = self.path + "MIND/MINDwhole_dev/news.tsv"

        f = open(news_path_whole_train, "w")
        h = open(news_path_whole_dev, "w")
        with open(news_path_train, "r") as g:
            with open(news_path_dev, "r") as l:
                for line in g:
                    f.write(line)
                for line in l:
                    f.write(line)

        with open(news_path_dev, "r") as g:
            for line in g:
                h.write(line)

        f.close()
        h.close()


    def gather_same_user_impr(self):
        """
        gather the impression of the same user to one record
        """
        behav_path = self.path + "MIND/MINDlarge_dev/behaviors.tsv"

        logging.info("gathering behavior log of {}".format(behav_path))
        behaviors = defaultdict(list)

        with open(behav_path, "r", encoding="utf-8") as rd:
            for idx in rd:
                impr_index, uid, time, history, impr = idx.strip("\n").split("\t")
                behaviors[uid].append([impr_index, uid, time, history, impr])

        for k,v in behaviors.items():
            behaviors[k] = sorted(v,key=lambda x: datetime.strptime(x[2], "%m/%d/%Y %X %p"))

        behavs = []
        for k,v in behaviors.items():
            record = []
            imprs = []
            record.extend(v[0][:4])
            for i,behav in enumerate(v):
                imprs.append(behav[4])
            record.append(" ".join(imprs))
            behavs.append(record)

        with open(self.path + "MIND/MINDlarge_dev/behaviors-uni.tsv","w",encoding="utf-8") as f:
            for record in behavs:
                f.write('\t'.join(record) + '\n')


def mrr_score(y_true, y_score):
    """Computing mrr score metric.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: mrr scores.
    """
    # descending rank prediction score, get corresponding index of candidate news
    order = np.argsort(y_score)[::-1]
    # get ground truth for these indexes
    y_true = np.take(y_true, order)
    # check whether the prediction news with max score is the one being clicked
    # calculate the inverse of its index
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def hit_score(y_true, y_score, k=10):
    """Computing hit score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: hit score.
    """
    ground_truth = np.where(y_true == 1)[0]
    argsort = np.argsort(y_score)[::-1][:k]
    for idx in argsort:
        if idx in ground_truth:
            return 1
    return 0


def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def cal_metric(labels, preds, metrics):
    """Calculate metrics,such as auc, logloss.
    """
    res = {}
    for metric in metrics:
        if metric == "auc":
            auc = np.mean(
                [
                    roc_auc_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["auc"] = round(auc, 4)
        elif metric == "rmse":
            rmse = mean_squared_error(np.asarray(labels), np.asarray(preds))
            res["rmse"] = np.sqrt(round(rmse, 4))
        elif metric == "logloss":
            # avoid logloss nan
            preds = [max(min(p, 1.0 - 10e-12), 10e-12) for p in preds]
            logloss = log_loss(np.asarray(labels), np.asarray(preds))
            res["logloss"] = round(logloss, 4)
        elif metric == "acc":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            acc = accuracy_score(np.asarray(labels), pred)
            res["acc"] = round(acc, 4)
        elif metric == "f1":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            f1 = f1_score(np.asarray(labels), pred)
            res["f1"] = round(f1, 4)
        elif metric == "mean_mrr":
            mean_mrr = np.mean(
                [
                    mrr_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["mean_mrr"] = round(mean_mrr, 4)
        elif metric.startswith("ndcg"):  # format like:  ndcg@2;4;6;8
            ndcg_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                ndcg_list = [int(token) for token in ks[1].split(";")]
            for k in ndcg_list:
                ndcg_temp = np.mean(
                    [
                        ndcg_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["ndcg@{0}".format(k)] = round(ndcg_temp, 4)
        elif metric.startswith("hit"):  # format like:  hit@2;4;6;8
            hit_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                hit_list = [int(token) for token in ks[1].split(";")]
            for k in hit_list:
                hit_temp = np.mean(
                    [
                        hit_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["hit@{0}".format(k)] = round(hit_temp, 4)
        else:
            raise ValueError("not define this metric {0}".format(metric))
    return res