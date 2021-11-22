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
            parser.add_argument("-p", "--path", dest="path", type=str, default="../../../Data/", help="root path for large-scale reusable data")
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

            parser.add_argument("--descend_history", dest="descend_history", help="whether to order history by time in descending", action="store_true", default=False)
            parser.add_argument("--debias", dest="debias", help="whether to add a learnable bias to each candidate news's score", type=bool, default=True)

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
            parser.add_argument("-pl", "--pooler", dest="pooler", help="choose bert pooler", choices=["avg","attn","cls"], default="attn")
            parser.add_argument("-rk", "--ranker", dest="ranker", help="choose ranker", choices=["onepass","original","cnn","knrm"], default="original")

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
                sampler_train = DistributedSampler(dataset_train, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle)
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

        elif self.mode == "encode":
            file_directory = self.path + "MIND/MIND{}_dev/".format(self.scale)

            dataset = MIND_history(self, file_directory)
            sampler = None

            loader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=pin_memory,
                                    num_workers=num_workers, drop_last=False, shuffle=shuffle, sampler=sampler)

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


    def get_bert_for_load(self):
        """
        transfer unilm to bert
        """
        bert_map = {
            "bert": "bert-base-uncased",
            "deberta": "microsoft/deberta-base",
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
        behav_paths = [self.path + "MIND/MINDlarge_dev/behaviors.tsv", self.path + "MIND/MINDsmall_dev/behaviors.tsv", self.path + "MIND/MINDdemo_dev/behaviors.tsv"]

        for behav_path in behav_paths:
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

            with open(behav_path,"w",encoding="utf-8") as f:
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