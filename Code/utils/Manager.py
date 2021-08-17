import torch
import re
import os
import datetime
import logging
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.stats as ss

from tqdm import tqdm
from typing import OrderedDict
from itertools import chain
from collections import defaultdict
from transformers import get_linear_schedule_with_warmup
# from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, accuracy_score, f1_score

import torch.distributed as dist
# from contextlib import contextmanager

logger = logging.getLogger(__name__)

hparam_list = ["name", "scale", "his_size", "k", "threshold", "lr", "bert-lr", "signal_length", "title_length", "abs_length"]


class Manager():
    """
    wrap training/evaluating processes
    """
    def __init__(self, args):
        for k,v in vars(args).items():
            if not k.startswith('__'):
                setattr(self, k, v)

    def save(self, model, epoch, step, optimizer=None, scheduler=None):
        """
            shortcut for saving the model and optimizer
        """

        save_path = "data/model_params/{}/{}_epoch{}_step{}_[k={}].model".format(
            self.name, self.scale, epoch, step, self.k)

        state_dict = model.state_dict()

        if re.search("pipeline", self.name):
            state_dict = {k: v for k, v in state_dict.items() if k not in [
                "encoder.news_repr.weight", "encoder.news_embedding.weight"]}

        save_dict = {}
        save_dict["model"] = state_dict
        save_dict["optimizer"] = optimizer
        save_dict["scheduler"] = scheduler

        torch.save(save_dict, save_path)
        logger.info("saved model of step {}, epoch {} at {}".format(
            step, epoch, save_path))


    def load(self, model, epoch, step, optimizer=None, scheduler=None):
        """
            shortcut for loading model and optimizer parameters
        """

        save_path = "data/model_params/{}/{}_epoch{}_step{}_[k={}].model".format(
            self.name, self.scale, epoch, step, self.k)

        state_dict = torch.load(save_path, map_location=torch.device(model.device))

        if self.world_size <= 1:
            # in case we load a DDP model checkpoint to a non-DDP model
            model_dict = OrderedDict()
            pattern = re.compile('module.')

            for k,v in state_dict['model'].items():
                if re.search("module", k):
                    model_dict[re.sub(pattern, '', k)] = v
                else:
                    model_dict = state_dict['model']
        else:
            model_dict = state_dict['model']

        if re.search("pipeline",self.name):
            logger.info("loading in pipeline")
            model.load_state_dict(model_dict, strict=False)
        else:
            model.load_state_dict(model_dict)

        if optimizer:
            optimizer.load_state_dict(state_dict["optimizer"])
        if scheduler:
            scheduler.load_state_dict(state_dict["scheduler"])

        logger.info("Loading model from {}...".format(save_path))


    def _log(self, res):
        """
            wrap logging, skip logging results on MINDdemo
        """

        # logger.info("evaluation results:{}".format(res))
        with open("performance.log", "a+") as f:
            d = {"name": self.name}
            for k, v in vars(self).items():
                if k in hparam_list:
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

        base_params = [v.parameters() for k,v in model.named_children() if k not in ['embedding','interactor']]
        bert_params = []


        if self.embedding == 'bert':
            bert_params.append(model.embedding.parameters())
        else:
            base_params.append(model.embedding.parameters())
        if self.interactor == 'bert':
            bert_params.append(model.interactor.parameters())
        else:
            base_params.append(model.interactor.parameters())


        optimizer = optim.Adam([
            {
                'params': chain(*bert_params),
                'lr': self.bert_lr #lr_schedule(args.pretrain_lr, 1, args)
            },
            {
                'params': chain(*base_params),
                'lr': self.lr #lr_schedule(args.lr, 1, args)
            }
        ])

        if self.scheduler == 'linear':
            total_steps = loader_train_length * self.epochs
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = self.warmup, # Default value in run_glue.py
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
    def _eval(self, model, dataloader, smoothing=1):
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

        for x in tqdm(dataloader, smoothing=smoothing):
            impr_indexes.extend(x["impression_index"].tolist())
            preds.extend(model(x).tolist())
            labels.extend(x["label"].tolist())


        # all_preds = []
        # all_labels = []
        # tmp_preds = []
        # tmp_labels = []
        # prev_impr = 1
        # for label, pred, impr_index in zip(labels, preds, impr_indexes):
        #     if impr_index != prev_impr:
        #         all_labels.append(tmp_labels.copy())
        #         all_preds.append(tmp_preds.copy())
        #         tmp_labels.clear()
        #         tmp_preds.clear()
        #         prev_impr += 1

        #     tmp_labels.extend(label)
        #     tmp_preds.extend(pred)

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
            self.load(model, self.epochs, self.step[0], optimizer, scheduler)

        if self.rank in [0,-1]:
            logger.info("evaluating...")

        # protect non-master node to return an object
        res = None

        impr_label_preds = self._eval(model, dataloader)

        # collect result across gpus when distributed evaluating
        if self.world_size > 1:
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

            logger.info("evaluation result of {} is {}".format(self.name, res))

            if log and self.rank in [0, -1]:
                res["epoch"] = self.epochs
                res["step"] = self.step[0]

                self._log(res)

        return res


    def _train(self, model, dataloader, optimizer, loss_func, epochs, scheduler=None, writer=None, interval=100, save_step=None, distributed=False):
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
        total_loss = 0
        total_steps = 0

        for epoch in range(epochs):
            epoch_loss = 0
            if distributed:
                dataloader.sampler.set_epoch(epoch)
            tqdm_ = tqdm(dataloader)

            for step, x in enumerate(tqdm_):

                optimizer.zero_grad(set_to_none=True)

                pred = model(x)
                label = x['label'].to(model.device)

                loss = loss_func(pred, label)

                epoch_loss += loss
                total_loss += loss

                loss.backward()

                optimizer.step()

                if scheduler:
                    scheduler.step()

                if step % interval == 0:
                    tqdm_.set_description(
                        "epoch {:d} , step {:d} , loss: {:.4f}".format(epoch+1, step, epoch_loss / step))
                    if writer:
                        for name, param in model.named_parameters():
                            writer.add_histogram(name, param, step)

                        writer.add_scalar("data_loss",
                                        total_loss/total_steps)

                if save_step:
                    if step % save_step == 0 and step > 0:
                        self.save(model, epoch+1, step, optimizer)

                total_steps += 1

            if writer:
                writer.add_scalar("epoch_loss", epoch_loss, epoch)

            self.save(model, epoch+1, 0, optimizer)


    def train(self, model, loaders, tb=False):
        """ wrap training process

        Args:
            model(torch.nn.Module): the model to be trained
            loaders(list): list of torch.utils.data.DataLoader
            en: shell parameter
        """
        model.train()
        writer = None

        if tb:
            writer = SummaryWriter("data/tb/{}/{}/{}/".format(
                self.name, self.scale, datetime.now().strftime("%Y%m%d-%H")))

        # in case the folder does not exists, create one
        os.makedirs("data/model_params/{}".format(self.name), exist_ok=True)

        logger.info("training...")
        loss_func = self._get_loss(model)
        optimizer, scheduler = self._get_optim(model, len(loaders[0]))

        self._train(model, loaders[0], optimizer, loss_func, self.epochs, scheduler=scheduler,
                        writer=writer, interval=self.interval, save_step=self.step[0], distributed=(self.world_size > 1))


    def _tune(self, model, loaders, optimizer, loss_func, scheduler=None, writer=None, interval=100, save_step=None, distributed=False):
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
        total_loss = 0
        total_steps = 0

        best_res = {"auc":0}

        for epoch in range(self.epochs):
            epoch_loss = 0
            if distributed:
                loaders[0].sampler.set_epoch(epoch)
            tqdm_ = tqdm(loaders[0], smoothing=0)

            for step, x in enumerate(tqdm_):

                optimizer.zero_grad(set_to_none=True)

                pred = model(x)
                label = x['label'].to(model.device)

                loss = loss_func(pred, label)

                epoch_loss += loss
                total_loss += loss

                loss.backward()

                optimizer.step()

                if scheduler:
                    scheduler.step()

                if step % interval == 0:
                    tqdm_.set_description(
                        "epoch {:d} , step {:d} , loss: {:.4f}".format(epoch+1, step, epoch_loss / step))
                    if writer:
                        for name, param in model.named_parameters():
                            writer.add_histogram(name, param, step)

                        writer.add_scalar("data_loss",
                                        total_loss/total_steps)

                if step % save_step == 0 and step > 0:
                    print("\n")
                    with torch.no_grad():
                        result = self.evaluate(model, loaders[1], log=False)

                        if self.rank in [0,-1]:
                            result["epoch"] = epoch+1
                            result["step"] = step

                            if result["auc"] > best_res["auc"]:
                                best_res = result
                                self.save(model, epoch+1, step, optimizer)
                                self._log(result)
                    # continue training
                    model.train()

            if writer:
                writer.add_scalar("epoch_loss", epoch_loss, epoch)

        return best_res


    def tune(self, model, loaders):
        """ train and evaluate sequentially

        Args:
            model(torch.nn.Module): the model to be trained
            loaders(list): list of torch.utils.data.DataLoader
            en: shell parameter
        """

        model.train()
        writer = None

        if self.tb:
            writer = SummaryWriter("data/tb/{}/{}/{}/".format(
                self.name, self.scale, datetime.now().strftime("%Y%m%d-%H")))

        # in case the folder does not exists, create one
        os.makedirs("data/model_params/{}".format(self.name), exist_ok=True)

        if self.rank in [-1, 0]:
            logger.info("training...")
        loss_func = self._get_loss(model)
        optimizer, scheduler = self._get_optim(model, len(loaders[0]))

        res = self._tune(model, loaders, optimizer, loss_func, scheduler=scheduler,
                        writer=writer, interval=self.interval, save_step=int(len(loaders[0])/self.val_freq - 1), distributed=(self.world_size > 1))

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

        self.load(model, self.epochs, self.step[0])

        if self.rank in [0,-1]:
            logger.info("testing...")

        impr_indexes = []
        preds = []
        for x in tqdm(loader_test, smoothing=0):
            impr_indexes.extend(x["impression_index"])
            preds.extend(model(x).tolist())

        if self.world_size > 1:
            outputs = [None for i in range(self.world_size)]
            dist.all_gather_object(outputs, (impr_indexes, preds))

            if self.rank == 0:
                impr_indexes = []
                preds = []
                for output in outputs:
                    impr_indexes.extend(output[0])
                    preds.extend(output[1])

                preds = self._group_lists(impr_indexes, preds)
        else:
            preds = self._group_lists(impr_indexes, preds)


        if self.rank in [0, -1]:
            save_directory = "data/results/{}".format(self.name)
            os.makedirs(save_directory, exist_ok=True)

            save_path = save_directory + "/{}_epoch{}_step{}_[k={}].txt".format(
                self.scale, self.epochs, self.step[0], self.k)

            index = 1
            with open(save_path, 'w') as f:
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
        for x in tqdm(loader):
            news = x['cdd_encoded_index'].long()
            news_embedding = model.embedding(news)
            news_repr,_ = model.encoder(news_embedding)

            news_reprs[start_pos:start_pos+model.batch_size] = news_repr
            news_embeddings[start_pos:start_pos+self.batch_size] = news_embedding

        os.makedirs('data/tensors/news_reprs/{}/'.format(self.name), exist_ok=True)
        os.makedirs('data/tensors/news_embeddings/{}/'.format(self.name), exist_ok=True)

        torch.save(news_reprs, "data/tensors/news_reprs/{}/{}_{}.tensor".format(
            self.name, scale, mode))
        torch.save(news_embeddings, "data/tensors/news_embeddings/{}/{}_{}.tensor".format(
            self.name, scale, mode))
        del news_reprs
        del news_embeddings
        logging.info("encoded news saved in data/tensors/news_**_{}_{}-[{}].tensor".format(scale, mode, self.name))


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