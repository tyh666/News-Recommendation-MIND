import torch
import re
import os
import logging
import argparse
import json
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.stats as ss
import torch.distributed as dist

from tqdm.auto import tqdm
from typing import OrderedDict
from collections import defaultdict
from transformers import get_linear_schedule_with_warmup
# from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, accuracy_score, f1_score
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler


logger = logging.getLogger(__name__)

hparam_list = ["epochs", "device", "path", "title_length", "step", "checkpoint", "smoothing", "num_workers", "pin_memory", "interval", "npratio", "metrics", "aggregator", "head_num", "rank"]

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
            parser.add_argument("-s", "--scale", dest="scale", help="data scale",
                                choices=["demo", "small", "large", "whole"], required=True)
            parser.add_argument("-m", "--mode", dest="mode", help="train or test",
                                choices=["train", "dev", "test", "tune", "encode", "inspect"], default="tune")
            parser.add_argument("-e", "--epochs", dest="epochs",
                                help="epochs to train the model", type=int, default=10)
            parser.add_argument("-d","--device", dest="device",
                                help="device to run on, -1 means cpu", choices=[i for i in range(-1,10)], type=int, default=0)
            parser.add_argument("-p", "--path", dest="path", type=str, default="../../../Data/", help="root path for large-scale reusable data")

            parser.add_argument("-bs", "--batch_size", dest="batch_size",
                                help="batch size", type=int, default=25)
            parser.add_argument("-hs", "--his_size", dest="his_size",
                                help="history size", type=int, default=50)
            parser.add_argument("-is", "--impr_size", dest="impr_size",
                                help="impression size for evaluating", type=int, default=50)
            # parser.add_argument("-tl", "--title_length", dest="title_length",
            #                     help="news title size", type=int, default=20)
            # parser.add_argument("-as", "--abs_length", dest="abs_length",
            #                     help="news abstract length", type=int, default=40)
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
                                help="learning rate of bert based modules", type=float, default=1e-5)
            parser.add_argument("-sm", "--smoothing", dest="smoothing", help="smoothing factor of tqdm", type=float, default=0.3)

            parser.add_argument("-div", "--diversify", dest="diversify", help="whether to diversify selection with news representation", action="store_true", default=False)
            parser.add_argument("--ascend_history", dest="ascend_history", help="whether to order history by time in ascending", action="store_true", default=False)
            parser.add_argument("--save_pos", dest="save_pos", help="whether to save token positions", action="store_true", default=False)
            parser.add_argument("--sep_his", dest="sep_his", help="whether to separate personalized terms from different news with an extra token", action="store_true", default=False)
            parser.add_argument("--full_attn", dest="full_attn", help="whether to interact among personalized terms (only in one-pass bert models)", action="store_true", default=False)
            parser.add_argument("--debias", dest="debias", help="whether to add a learnable bias to each candidate news's score", action="store_true", default=False)
            parser.add_argument("--no_dedup", dest="no_dedup", help="whether to deduplicate tokens", action="store_true", default=False)
            parser.add_argument("--no_rm_punc", dest="no_rm_punc", help="whether to mask punctuations when selecting", action="store_true", default=False)
            parser.add_argument("--no_order_embed", dest="no_order_embed", help="whether to add an extra embedding to ps terms from the same historical news", action="store_true", default=False)

            parser.add_argument("--num_workers", dest="num_workers", help="worker number of a dataloader", type=int, default=0)
            parser.add_argument("--shuffle", dest="shuffle", help="whether to shuffle the indices", action="store_true", default=False)
            parser.add_argument("--pin_memory", dest="pin_memory", help="whether to pin memory to speed up tensor transfer", action="store_true", default=False)
            parser.add_argument("--scheduler", dest="scheduler", help="choose schedule scheme for optimizer", choices=["linear","none"], default="linear")
            parser.add_argument("--warmup", dest="warmup", help="warmup steps of scheduler", type=int, default=10000)
            parser.add_argument("--interval", dest="interval", help="the step interval to update processing bar", default=10, type=int)

            parser.add_argument("--npratio", dest="npratio", help="the number of unclicked news to sample when training", type=int, default=4)
            parser.add_argument("--metrics", dest="metrics", help="metrics for evaluating the model", type=str, default="")

            parser.add_argument("-g", "--granularity", dest="granularity", help="the granularity for reduction", choices=["token", "avg", "first", "sum"], default="avg")
            parser.add_argument("-emb", "--embedding", dest="embedding", help="choose embedding", choices=["bert","random","deberta"], default="bert")
            parser.add_argument("-encn", "--encoderN", dest="encoderN", help="choose news encoder", choices=["cnn","rnn","npa","fim","mha","bert"], default="cnn")
            parser.add_argument("-encu", "--encoderU", dest="encoderU", help="choose user encoder", choices=["avg","attn","cnn","rnn","lstur","mha"], default="rnn")
            parser.add_argument("-slc", "--selector", dest="selector", help="choose history selector", choices=["recent","sfi"], default="sfi")
            parser.add_argument("-red", "--reducer", dest="reducer", help="choose document reducer", choices=["bm25","matching","bow","entity","first"], default="matching")
            parser.add_argument("-fus", "--fuser", dest="fuser", help="choose term fuser", choices=["union"], default="union")
            parser.add_argument("-rk", "--ranker", dest="ranker", help="choose ranker", choices=["onepass","original","cnn","knrm"], default="onepass")
            parser.add_argument("-agg", "--aggregator", dest="aggregator", help="choose history aggregator, only used in TTMS", choices=["avg","attn","cnn","rnn","lstur","mha"], default=None)

            parser.add_argument("-k", dest="k", help="the number of the terms to extract from each news article", type=int, default=5)
            parser.add_argument("-thr", "--threshold", dest="threshold", help="threshold to mask terms", default=-float("inf"), type=float)
            parser.add_argument("--bert", dest="bert", help="choose bert model", choices=["bert-base-uncased"], default="bert-base-uncased")

            parser.add_argument("--tb", dest="tb", action="store_true", default=False)
            parser.add_argument("-sd","--seed", dest="seed", default=42, type=int)

            parser.add_argument("-hn", "--head_num", dest="head_num", help="number of multi-heads", type=int, default=12)
            parser.add_argument("-ws", "--world_size", dest="world_size", help="total number of gpus", default=0, type=int)
            parser.add_argument("-br", "--base_rank", dest="base_rank", help="modify port when launching multiple tasks on one node", default=0, type=int)

            args = parser.parse_args()

            args.cdd_size = args.npratio + 1
            args.metrics = "auc,mean_mrr,ndcg@5,ndcg@10".split(",") + [i for i in args.metrics.split(",") if i]
            if args.device == -1:
                args.device = "cpu"
                args.pin_memory = False

            if args.embedding == 'deberta':
                args.bert = 'microsoft/deberta-base'

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
        return "\n" + json.dumps({k:v for k,v in vars(self).items() if k not in hparam_list}, sort_keys=False, indent=4)


    def setup(self, rank):
        """
        set up distributed training and fix seeds
        """
        if self.world_size > 1:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(12355 + self.base_rank)

            # initialize the process group
            dist.init_process_group("nccl", rank=rank, world_size=self.world_size)

            os.environ["TOKENIZERS_PARALLELISM"] = "True"
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


    def cleanup(self):
        """
        shut down the distributed training process
        """
        if self.world_size > 1:
            dist.destroy_process_group()


    def prepare(self):
        """ prepare dataloader and several paths

        Returns:
            loaders(list of dataloaders): 0-loader_train/test/dev, 1-loader_dev, 2-loader_validate
        """
        from .MIND import MIND
        from .utils import Partition_Sampler

        if self.rank in [-1, 0]:
            logger.info("Hyper Parameters are {}".format(self))
            logger.info("preparing dataset...")

        mind_path = self.path + "MIND"
        shuffle = self.shuffle
        pin_memory = self.pin_memory
        num_workers = self.num_workers

        if self.mode in ["train", "tune"]:
            news_file_train = mind_path+"/MIND"+self.scale+"_train/news.tsv"
            behavior_file_train = mind_path+"/MIND" + \
                self.scale+"_train/behaviors.tsv"
            news_file_dev = mind_path+"/MIND"+self.scale+"_dev/news.tsv"
            behavior_file_dev = mind_path+"/MIND"+self.scale+"_dev/behaviors.tsv"

            dataset_train = MIND(config=self, news_file=news_file_train,
                            behaviors_file=behavior_file_train)
            dataset_dev = MIND(config=self, news_file=news_file_dev,
                            behaviors_file=behavior_file_dev)

            # FIXME: multi view dataset

            if self.world_size > 0:
                sampler_train = DistributedSampler(dataset_train, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle)
                sampler_dev = Partition_Sampler(dataset_dev, num_replicas=self.world_size, rank=self.rank)
            else:
                sampler_train = None
                sampler_dev = None
            loader_train = DataLoader(dataset_train, batch_size=self.batch_size, pin_memory=pin_memory,
                                    num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler_train)
            loader_dev = DataLoader(dataset_dev, batch_size=1, pin_memory=pin_memory,
                                    num_workers=num_workers, drop_last=False, sampler=sampler_dev)

            return loader_train, loader_dev

        elif self.mode in ["dev", "inspect"]:
            news_file_dev = mind_path+"/MIND"+self.scale+"_dev/news.tsv"
            behavior_file_dev = mind_path+"/MIND"+self.scale+"_dev/behaviors.tsv"

            dataset_dev = MIND(config=self, news_file=news_file_dev,
                                behaviors_file=behavior_file_dev)

            if self.world_size > 0:
                sampler_dev = Partition_Sampler(dataset_dev, num_replicas=self.world_size, rank=self.rank)
            else:
                sampler_dev = None
            loader_dev = DataLoader(dataset_dev, batch_size=1, pin_memory=pin_memory,
                                    num_workers=num_workers, drop_last=False, sampler=sampler_dev)

            return loader_dev,

        elif self.mode == "test":
            news_file_test = mind_path+"/MIND"+self.scale+"_test/news.tsv"
            behavior_file_test = mind_path+"/MIND"+self.scale+"_test/behaviors.tsv"

            dataset_test = MIND(self, news_file_test, behavior_file_test)

            # FIXME distributed test
            if self.world_size > 0:
                sampler_test = Partition_Sampler(dataset_test, num_replicas=self.world_size, rank=self.rank)
            else:
                sampler_test = None
            loader_test = DataLoader(dataset_test, batch_size=1, pin_memory=pin_memory,
                                    num_workers=num_workers, drop_last=False, sampler=sampler_test)

            return loader_test,


    def save(self, model, step, optimizer=None):
        """
            shortcut for saving the model and optimizer
        """
        save_path = "data/model_params/{}/{}_step{}_[k={}].model".format(
            self.name, self.scale, step, self.k)

        logger.info("saving model at {}...".format(save_path))

        model_dict = model.state_dict()

        save_dict = {}
        save_dict["model"] = model_dict
        save_dict["optimizer"] = optimizer.state_dict()

        torch.save(save_dict, save_path)


    def load(self, model, step, optimizer=None):
        """
            shortcut for loading model and optimizer parameters
        """

        save_path = "data/model_params/{}/{}_step{}_[k={}].model".format(
            self.name, self.scale, step, self.k)

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

        if re.search("pipeline",self.name):
            logger.info("loading in pipeline")
            model.load_state_dict(model_dict, strict=False)
        else:
            model.load_state_dict(model_dict)

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

            f.write(str(d)+"\n")
            f.write(str(res) + "\n")
            f.write("\n")


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


    def _group_lists(self, impr_indexes, *associated_lists):
        """
            group lists by impr_index
        Args:
            associated_lists: list of lists, where list[i] is associated with the impr_indexes[i]

        Returns:
            all_labels: grouped labels
            all_preds: grouped preds
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


    @torch.no_grad()
    def _eval(self, model, dataloader):
        """ making prediction and gather results into groups according to impression_id

        Args:
            dataloader(torch.utils.data.DataLoader): provide data

        Returns:
            impr_indexes: impression ids
            labels: labels
            preds: preds
        """

        # FIXME: need to modify when distributed testing
        impr_indexes = []
        labels = []
        preds = []

        max_input = {
            "cdd_id": torch.empty(1, self.impr_size).random_(0,10),
            "his_id": torch.empty(1, self.his_size).random_(0,10),
            "cdd_encoded_index": torch.empty(1, self.impr_size, self.signal_length).random_(0,10),
            "his_encoded_index": torch.empty(1, self.his_size, self.signal_length).random_(0,10),
            "cdd_attn_mask": torch.ones(1, self.impr_size, self.signal_length),
            "his_attn_mask": torch.ones(1, self.his_size, self.signal_length),
            "cdd_subword_index": torch.ones(1, self.impr_size, self.signal_length, 2, dtype=torch.int64),
            "his_subword_index": torch.ones(1, self.his_size, self.signal_length, 2, dtype=torch.int64),
            "cdd_mask": torch.ones((1, self.impr_size, 1)),
            "his_mask": torch.ones((1, self.his_size, 1)),
        }
        if self.reducer == "matching":
            max_input["his_refined_mask"] = torch.ones(1, self.his_size, self.signal_length)
        elif self.reducer == "bow":
            max_input["cdd_encoded_index"] = torch.rand(1, self.impr_size, self.signal_length, 2).random_(0,10)
            max_input["his_encoded_index"] = torch.rand(1, self.his_size, self.signal_length, 2).random_(0,10)
            max_input["his_refined_mask"] = max_input["his_attn_mask"]
        elif self.reducer in ["bm25", "entity", "first"]:
            max_input["his_encoded_index"] = max_input["his_encoded_index"][:, :, :self.k+1]
            max_input["his_attn_mask"] = max_input["his_attn_mask"][:, :, :self.k+1]
            max_input["his_subword_index"] = max_input["his_subword_index"][:, :, :self.k+1]

        model(max_input)

        for x in tqdm(dataloader, smoothing=self.smoothing, ncols=150, leave=True):
            impr_indexes.extend(x["impr_index"].tolist())
            preds.extend(model(x)[0].tolist())
            labels.extend(x["label"].tolist())

        return impr_indexes, labels, preds


    def evaluate(self, model, dataloader, load=False, log=True, optimizer=None, scheduler=None):
        """Evaluate the given file and returns some evaluation metrics.

        Args:
            self(nn.Module)
            self(dict)
            dataloader(torch.utils.data.DataLoader)
            loading(bool): whether to load model
            log(bool): whether to log

        Returns:
            res(dict): A dictionary contains evaluation metrics.
        """
        # multiple evaluation steps

        model.eval()

        if load:
            self.load(model, self.checkpoint, optimizer)

        if self.rank in [0,-1]:
            logger.info("evaluating...")

        # protect non-master node to return an object
        res = None

        impr_label_preds = self._eval(model, dataloader)

        # collect result across gpus when distributed evaluating
        if self.world_size > 1:
            dist.barrier(device_ids=[self.rank])
            outputs = [None for i in range(self.world_size)]
            dist.all_gather_object(outputs, impr_label_preds)

            if self.rank == 0:
                impr_indexes = []
                labels = []
                preds = []
                for output in outputs:
                    impr_indexes.extend(output[0])
                    labels.extend(output[1])
                    preds.extend(output[2])

                labels, preds = self._group_lists(impr_indexes, labels, preds)

        else:
            labels, preds = self._group_lists(*impr_label_preds)


        if self.rank in [0, -1]:
            res = cal_metric(labels, preds, self.metrics)

            logger.info("\nevaluation result of {} is {}".format(self.name, res))

            if log and self.rank in [0, -1]:
                res["epoch"] = self.epochs
                res["step"] = self.step
                self._log(res)

        return res


    def _train(self, model, dataloader, optimizer, loss_func, epochs, scheduler=None):
        """ train model and print loss meanwhile
        Args:
            dataloader(torch.utils.data.DataLoader): provide data
            optimizer(list of torch.nn.optim): optimizer for training
            loss_func(torch.nn.Loss): loss function for training
            epochs(int): training epochs
            schedulers
            writer(torch.utils.tensorboard.SummaryWriter): tensorboard writer
            interval(int): within each epoch, the interval of training steps to display loss
            save_step(int): how many steps to save the model
        Returns:
            model: trained model
        """
        steps = 0
        interval = self.interval
        save_step = self.step
        distributed = self.world_size > 1

        if self.rank in [0, -1]:
            logger.info("training...{}".format(self.name))
            logger.info("total training step: {}".format(self.epochs * len(dataloader)))

        # if self.tb:
        #     writer = SummaryWriter("data/tb/{}/{}/{}/".format(
        #         self.name, self.scale, datetime.now().strftime("%Y%m%d-%H")))

        for epoch in range(epochs):
            epoch_loss = 0
            if distributed:
                dataloader.sampler.set_epoch(epoch)
            tqdm_ = tqdm(dataloader, smoothing=self.smoothing, ncols=150, leave=True)

            for step, x in enumerate(tqdm_):

                optimizer.zero_grad(set_to_none=True)

                pred = model(x)[0]
                label = x["label"].to(model.device)

                loss = loss_func(pred, label)

                epoch_loss += float(loss)

                loss.backward()

                optimizer.step()

                if scheduler:
                    scheduler.step()

                if step % interval == 0:
                    tqdm_.set_description(
                        "epoch: {:d}  step: {:d} total_step: {:d}  loss: {:.4f}".format(epoch + 1, step, steps, epoch_loss / step))
                    # if writer:
                    #     for name, param in model.named_parameters():
                    #         writer.add_histogram(name, param, step)

                    #     writer.add_scalar("data_loss",
                    #                     total_loss/total_steps)

                if save_step:
                    if steps % save_step == 0 and steps > 0:
                        self.save(model, steps, optimizer)

                steps += 1


    def train(self, model, loaders, tb=False):
        """ wrap training process

        Args:
            model(torch.nn.Module): the model to be trained
            loaders(list): list of torch.utils.data.DataLoader
            en: shell parameter
        """
        model.train()

        if self.rank in [0,-1]:
            # in case the folder does not exists, create one
            os.makedirs("data/model_params/{}".format(self.name), exist_ok=True)

        loss_func = self._get_loss(model)
        optimizer, scheduler = self._get_optim(model, len(loaders[0]))

        if self.checkpoint:
            self.load(model, self.checkpoint, optimizer)

        self._train(model, loaders[0], optimizer, loss_func, self.epochs, scheduler=scheduler)


    def _tune(self, model, loaders, optimizer, loss_func, scheduler=None):
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

        if self.scale == "demo":
            save_step = len(loaders[0]) - 1
        else:
            save_step = self.step

        distributed = self.world_size > 1
        # if self.tb:
        #     writer = SummaryWriter("data/tb/{}/{}/{}/".format(
        #         self.name, self.scale, datetime.now().strftime("%Y%m%d-%H")))

        best_res = {"auc":0}

        if self.rank in [0, -1]:
            logger.info("tuning {}...".format(self.name))
            logger.info("total training step: {}".format(self.epochs * len(loaders[0])))

        for epoch in range(self.epochs):
            epoch_loss = 0
            if distributed:
                loaders[0].sampler.set_epoch(epoch)
            tqdm_ = tqdm(loaders[0], smoothing=self.smoothing, ncols=150, leave=True)

            for step, x in enumerate(tqdm_):

                optimizer.zero_grad(set_to_none=True)

                pred = model(x)[0]
                label = x["label"].to(model.device)

                loss = loss_func(pred, label)
                epoch_loss += loss

                loss.backward()

                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                optimizer.step()

                if scheduler:
                    scheduler.step()

                if step % interval == 0:
                    tqdm_.set_description(
                        "epoch: {:d},  step: {:d},  total_step: {:d},  loss: {:.4f}".format(epoch + 1, step, steps, epoch_loss / step))
                    # if writer:
                    #     for name, param in model.named_parameters():
                    #         writer.add_histogram(name, param, step)

                    #     writer.add_scalar("data_loss",
                    #                     total_loss/total_steps)

                if steps % save_step == 0 and steps > 0:
                    print("\n")
                    with torch.no_grad():
                        result = self.evaluate(model, loaders[1], log=False)

                        if self.rank in [0,-1]:
                            result["step"] = steps

                            if result["auc"] > best_res["auc"]:
                                best_res = result
                                self.save(model, steps, optimizer)
                                self._log(result)
                    # continue training
                    model.train()

                steps += 1

        return best_res


    def tune(self, model, loaders):
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

        res = self._tune(model, loaders, optimizer, loss_func, scheduler=scheduler)

        if self.rank in [-1,0]:
            logger.info("Best result: {}".format(res))
            self._log(res)


    @torch.no_grad()
    def test(self, model, loader_test):
        """ test the model on test dataset of MINDlarge

        Args:
            model
            loader_test: DataLoader of MINDlarge_test
        """
        model.eval()

        self.load(model, self.checkpoint)

        if self.rank in [0,-1]:
            logger.info("testing...")

        max_input = {
            "cdd_id": torch.empty(1, self.impr_size).random_(0,10),
            "his_id": torch.empty(1, self.his_size).random_(0,10),
            "cdd_encoded_index": torch.empty(1, self.impr_size, self.signal_length).random_(0,10),
            "his_encoded_index": torch.empty(1, self.his_size, self.signal_length).random_(0,10),
            "cdd_attn_mask": torch.ones(1, self.impr_size, self.signal_length),
            "his_attn_mask": torch.ones(1, self.his_size, self.signal_length),
            "cdd_subword_index": torch.ones(1, self.impr_size, self.signal_length, 2, dtype=torch.int64),
            "his_subword_index": torch.ones(1, self.his_size, self.signal_length, 2, dtype=torch.int64),
            "cdd_mask": torch.ones((1, self.impr_size, 1)),
            "his_mask": torch.ones((1, self.his_size, 1)),
        }
        if self.reducer == "matching":
            max_input["his_refined_mask"] = torch.ones(1, self.his_size, self.signal_length)
        elif self.reducer == "bow":
            max_input["cdd_encoded_index"] = torch.rand(1, self.impr_size, self.signal_length, 2).random_(0,10)
            max_input["his_encoded_index"] = torch.rand(1, self.his_size, self.signal_length, 2).random_(0,10)
            max_input["his_refined_mask"] = max_input["his_attn_mask"]
        elif self.reducer in ["bm25", "entity", "first"]:
            max_input["his_encoded_index"] = max_input["his_encoded_index"][:, :, :self.k+1]
            max_input["his_attn_mask"] = max_input["his_attn_mask"][:, :, :self.k+1]
            max_input["his_subword_index"] = max_input["his_subword_index"][:, :, :self.k+1]

        model(max_input)

        impr_indexes = []
        preds = []
        for x in tqdm(loader_test, smoothing=self.smoothing, ncols=150, leave=True):
            impr_indexes.extend(x["impr_index"].tolist())
            preds.extend(model(x)[0].tolist())

        if self.world_size > 1:
            dist.barrier(device_ids=[self.rank])
            outputs = [None for i in range(self.world_size)]
            dist.all_gather_object(outputs, (impr_indexes, preds))

            if self.rank == 0:
                impr_indexes = []
                preds = []
                for output in outputs:
                    impr_indexes.extend(output[0])
                    preds.extend(output[1])

                preds = self._group_lists(impr_indexes, preds)[0]
        else:
            preds = self._group_lists(impr_indexes, preds)[0]

        if self.rank in [0, -1]:
            save_directory = "data/results/{}".format(self.name + "/{}_step{}_[k={}]".format(self.scale, self.checkpoint, self.k))
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
    def encode(self, model, loader):
        """
            Encode news of self.scale in self.mode
        """

        # very important
        model.eval()
        news_num_dict = {
            "demo": {
                "train": 51282,
                "dev": 42416
            },
            "small": {
                "train": 51282,
                "dev": 42416
            },
            "large": {
                "train": 101527,
                "dev": 72023,
                "test": 120961
            }
        }
        scale = loader.dataset.scale
        mode = loader.dataset.mode

        news_num = news_num_dict[scale][mode]

        news_reprs = torch.zeros((news_num + 1, model.hidden_dim))
        news_embeddings = torch.zeros(
            (news_num + 1, model.signal_length, model.embedding_dim))

        start_pos = 0
        for x in tqdm(loader, ncols=150, leave=True):
            news = x["cdd_encoded_index"].long()
            news_embedding = model.embedding(news)
            news_repr,_ = model.encoder(news_embedding)

            news_reprs[start_pos:start_pos+model.batch_size] = news_repr
            news_embeddings[start_pos:start_pos+self.batch_size] = news_embedding

        os.makedirs("data/tensors/news_reprs/{}/".format(self.name), exist_ok=True)
        os.makedirs("data/tensors/news_embeddings/{}/".format(self.name), exist_ok=True)

        torch.save(news_reprs, "data/tensors/news_reprs/{}/{}_{}.tensor".format(
            self.name, scale, mode))
        torch.save(news_embeddings, "data/tensors/news_embeddings/{}/{}_{}.tensor".format(
            self.name, scale, mode))
        del news_reprs
        del news_embeddings
        logging.info("encoded news saved in data/tensors/news_**_{}_{}-[{}].tensor".format(scale, mode, self.name))


    @torch.no_grad()
    def inspect(self, model, loader):
        """
        inspect personalized terms
        """
        import pickle
        from transformers import AutoTokenizer

        model.eval()
        logger.info("inspecting {}...".format(self.name))

        with open(loader.dataset.cache_directory + "news.pkl", "rb") as f:
            news = pickle.load(f)["encoded_news"][:, :self.signal_length]
        with open(loader.dataset.cache_directory + "bm25.pkl", "rb") as f:
            bm25_terms = pickle.load(f)["encoded_news"][:, :self.k + 1]
        with open(loader.dataset.cache_directory + "entity.pkl", "rb") as f:
            entities = pickle.load(f)["encoded_news"][:, :self.k + 1]

        self.load(model, self.checkpoint)
        t = AutoTokenizer.from_pretrained(self.bert)

        logger.info("press <ENTER> to continue")

        jumpout = False
        for x in loader:
            _, term_indexes = model(x)
            his_encoded_index = x["his_encoded_index"]
            if self.reducer == "bow":
                his_encoded_index = his_encoded_index[:, :, :, 0]
            his_id = x["his_id"]

            for i,batch in enumerate(his_encoded_index):
                for j,his_token_ids in enumerate(batch):
                    print("*******************************************************")
                    tokens = t.convert_ids_to_tokens(his_token_ids)
                    if self.granularity != "token":
                        terms = self.convert_tokens_to_words(tokens, punctuation=True)
                        terms.extend(["[PAD]"] * (self.signal_length - len(terms)))
                        terms = np.asarray(terms)
                    else:
                        if self.embedding == 'deberta':
                            terms = []
                            for token in tokens:
                                if token.startswith('Ġ'):
                                    terms.append(token[1:])
                                else:
                                    terms.append(token)
                        terms = np.asarray(terms)
                    ps_term_ids = term_indexes[i, j].cpu().numpy()
                    ps_terms = terms[ps_term_ids]

                    if ps_terms[0] == "[PAD]":
                        jumpout = True
                        break
                    else:
                        print("[personalized terms]\n\t {}".format(" ".join(ps_terms)))
                        print("[bm25 terms]\n\t {}".format(t.decode(bm25_terms[his_id[i,j]])))
                        print("[entities]\n\t {}".format(t.decode(entities[his_id[i,j]])))
                        print("[original news]\n\t {}".format(t.decode(news[his_id[i,j]][:self.signal_length], skip_special_tokens=True)))

                        command = input()
                        if command == "n":
                            break
                        elif command == "q":
                            return

                if jumpout:
                    jumpout = False
                    break


    def convert_tokens_to_words(self, tokens, punctuation=False):
        """
        wrapping bert/deberta
        """
        if self.embedding in ['bert','random']:
            from utils.utils import convert_tokens_to_words_bert
            return convert_tokens_to_words_bert(tokens)
        elif self.embedding == 'deberta':
            if punctuation:
                from utils.utils import convert_tokens_to_words_deberta_punctuation
                return convert_tokens_to_words_deberta_punctuation(tokens)
            else:
                from utils.utils import convert_tokens_to_words_deberta
                return convert_tokens_to_words_deberta(tokens)


    def get_special_token_id(self, token):
        special_token_map = {
            "bert-base-uncased":{
                "[CLS]": 101,
                "[SEP]": 102,
            },
            "microsoft/deberta-base":{
                "[CLS]": 1,
                "[SEP]": 2,
            }
        }
        return special_token_map[self.bert][token]


    def construct_nid2idx(self, scale=None, mode=None):
        """
            Construct news to newsID dictionary, index starting from 1
        """
        import pandas as pd
        import json
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

        h = open("data/dictionaries/nid2idx_{}_{}.json".format(self.scale, self.mode), "w")
        json.dump(nid2index, h, ensure_ascii=False)
        h.close()


    def construct_uid2idx(self, scale=None):
        """
            Construct user to userID dictionary, index starting from 1
        """
        import pandas as pd
        import json
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