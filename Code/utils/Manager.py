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
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, accuracy_score, f1_score

# import torch.distributed as dist
# from contextlib import contextmanager

logger = logging.getLogger(__name__)

hparam_list = ["name", "scale", "his_size", "k", "threshold", "learning_rate", "signal_length"]

# def dist_sync(func):
#     def void_dict():
#         return defaultdict(int)

#     def wrapper(*args, **kargs):
#         if args[0].rank == 0:
#             res = func(*args, **kargs)
#         else:
#             res = void_dict()

#         if args[0].world_size > 1:
#             print("fuck the barrier")
#             dist.barrier()
#             print("fuck the barrier after")
#         return res
#     return wrapper

# @contextmanager
# def only_on_main_process(local_rank):
#     """
#     Decorator to make all processes in distributed training wait for each local_master to do something.
#     """
#     need = True
#     if local_rank != 0:
#         need = False
#     yield need
#     # QUES: why still needs to barrier?
#     if local_rank == 0:
#         dist.barrier()

class Manager():
    """
    wrap training/evaluating processes
    """
    def __init__(self, args):
        for k,v in vars(args).items():
            if not k.startswith('__'):
                setattr(self, k, v)

        self.cdd_size = self.npratio + 1

    # @dist_sync
    def save(self, model, epoch, step, optimizers=[]):
        """
            shortcut for saving the model and optimizer
        """
        # FIXME: checkpoints

        # if self.scale == 'demo':
        #     return

        save_path = "data/model_params/{}/{}_epoch{}_step{}_[k={}].model".format(
            self.name, self.scale, epoch, step, self.k)

        state_dict = model.state_dict()

        if re.search("pipeline", self.name):
            state_dict = {k: v for k, v in state_dict.items() if k not in [
                "encoder.news_repr.weight", "encoder.news_embedding.weight"]}

        save_dict = {}
        save_dict["model"] = state_dict

        if len(optimizers) > 1:
            save_dict["optimizer"] = optimizers[0].state_dict()
            save_dict["optimizer_embedding"] = optimizers[1].state_dict()
        else:
            save_dict["optimizer"] = optimizers[0].state_dict()

        # if schedulers:
        #     if len(schedulers) > 1:
        #         save_dict["scheduler"] = schedulers[0].state_dict()
        #         save_dict["scheduler_embedding"] = schedulers[1].state_dict()
        #     else:
        #         save_dict["scheduler"] = schedulers[0].state_dict()

        torch.save(save_dict, save_path)
        logger.info("saved model of step {}, epoch {} at {}".format(
            step, epoch, save_path))


    # @dist_sync
    def load(self, model, epoch, step, optimizers=None):
        """
            shortcut for loading model and optimizer parameters
        """

        save_path = "data/model_params/{}/{}_epoch{}_step{}_[k={}].model".format(
            self.name, self.scale, epoch, step, self.k)

        state_dict = torch.load(save_path, map_location=model.device)
        if re.search("pipeline",self.name):
            logger.info("loading in pipeline")
            model.load_state_dict(state_dict["model"], strict=False)
        else:
            model.load_state_dict(state_dict["model"])

        if optimizers:
            optimizers[0].load_state_dict(state_dict["optimizer"])
            if len(optimizers) > 1:
                optimizers[1].load_state_dict(state_dict["optimizer_embedding"])

        # if schedulers:
        #     schedulers[0].load_state_dict(state_dict["scheduler"])
        #     if len(schedulers) > 1:
        #         schedulers[1].load_state_dict(state_dict["scheduler_embedding"])

        logger.info("Loading model from {}...".format(save_path))


    # @dist_sync
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
        if self.cdd_size > 1:
            loss = nn.NLLLoss()
        else:
            loss = nn.BCELoss()

        return loss


    def _get_optim(self, model, loader_train):
        """
            get optimizer/scheduler
        """

        learning_rate = self.learning_rate

        # if self.spadam:
        #     optimizer_param = optim.Adam(
        #         parameter(self, model, ["encoder.embedding.weight","encoder.user_embedding.weight"], exclude=True), lr=learning_rate)
        #     optimizer_embedding = optim.SparseAdam(
        #         list(parameter(self, model, ["encoder.embedding.weight","encoder.user_embedding.weight"])), lr=learning_rate)

        #     optimizers = (optimizer_param, optimizer_embedding)

        #     if self.schedule == "linear":
        #         scheduler_param =get_linear_schedule_with_warmup(optimizer_param, num_warmup_steps=0, num_training_steps=len(loader_train) * self.epochs)
        #         scheduler_embedding =get_linear_schedule_with_warmup(optimizer_embedding, num_warmup_steps=0, num_training_steps=len(loader_train) * self.epochs)
        #         schedulers = (scheduler_param, scheduler_embedding)
        #     else:
        #         schedulers = []

        # else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizers = (optimizer,)
        if self.schedule == "linear":
            scheduler =get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(loader_train) * self.epochs)
            schedulers = (scheduler,)
        else:
            schedulers = []

        # FIXME
        # if "checkpoint" in self:
        #     logger.info("loading checkpoint...")
        #     ck = self.checkpoint.split(",")
        #     # modify the epoch so the model can be properly saved
        #     self.load(self, model, ck[0], ck[1], optimizers)

        return optimizers, schedulers


    def _get_label(self, model, x):
        """
            parse labels to label indexes, used in NLLoss
        """
        if self.cdd_size > 1:
            index = torch.arange(0, self.cdd_size, device=model.device).expand(x['labels'].shape[0], -1)
            label = x["labels"] == 1
            label = index[label]
        else:
            label = x["labels"]

        return label


    @torch.no_grad()
    def _eval(self, model, dataloader, smoothing=1):
        """ making prediction and gather results into groups according to impression_id

        Args:
            dataloader(torch.utils.data.DataLoader): provide data

        Returns:
            all_impr_ids: impression ids after group
            all_labels: labels after group.
            all_preds: preds after group.
        """
        preds = []
        labels = []
        imp_indexes = []

        for batch_data_input in tqdm(dataloader, smoothing=smoothing):
            pred = model(batch_data_input).squeeze(dim=-1).tolist()
            preds.extend(pred)
            label = batch_data_input["labels"].squeeze(dim=-1).tolist()
            labels.extend(label)
            imp_indexes.extend(batch_data_input["impression_index"])

        all_keys = list(set(imp_indexes))
        all_keys.sort()
        group_labels = {k: [] for k in all_keys}
        group_preds = {k: [] for k in all_keys}

        for l, p, k in zip(labels, preds, imp_indexes):
            group_labels[k].append(l)
            group_preds[k].append(p)

        all_impr_ids = group_labels.keys()
        all_labels = list(group_labels.values())
        all_preds = list(group_preds.values())

        return all_impr_ids, all_labels, all_preds

    # @dist_sync
    def evaluate(self, model, dataloader, load=False, log=True):
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

        cdd_size = self.cdd_size
        model.cdd_size = 1
        model.eval()

        if load:
            self.load(model, self.epochs, self.step[0])

        logger.info("evaluating...")

        imp_indexes, labels, preds = self._eval(model, dataloader)

        res = cal_metric(labels, preds, self.metrics.split(","))

        if log:
            res["epoch"] = self.epochs
            res["step"] = self.step[0]

            self._log(model, res, self)

        model.train()
        model.cdd_size = cdd_size

        return res


    def _train(self, model, dataloader, optimizers, loss_func, epochs, schedulers=None, writer=None, interval=100, save_step=None):
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
            tqdm_ = tqdm(dataloader)
            for step, x in enumerate(tqdm_):

                for optimizer in optimizers:
                    optimizer.zero_grad()

                pred = model(x)
                label = self._get_label(model, x)

                loss = loss_func(pred, label)

                epoch_loss += loss
                total_loss += loss

                loss.backward()

                for optimizer in optimizers:
                    optimizer.step()

                if schedulers:
                    for scheduler in schedulers:
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
                        self.save(model, epoch+1, step, optimizers)

                total_steps += 1

            if writer:
                writer.add_scalar("epoch_loss", epoch_loss, epoch)

            self.save(epoch+1, 0, optimizers)


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
        optimizers, schedulers = self._get_optim(model, loaders[0])

        self._train(model, loaders[0], optimizers, loss_func, self.epochs, schedulers=schedulers,
                        writer=writer, interval=self.interval, save_step=self.step[0])


    def _tune(self, model, loaders, optimizers, loss_func, schedulers=[], writer=None, interval=100, save_step=None):
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
            tqdm_ = tqdm(loaders[0], smoothing=0)
            for step, x in enumerate(tqdm_):

                for optimizer in optimizers:
                    optimizer.zero_grad()

                pred = model(x)
                label = self._get_label(model, x)

                loss = loss_func(pred, label)

                epoch_loss += loss
                total_loss += loss

                loss.backward()

                for optimizer in optimizers:
                    optimizer.step()

                if schedulers:
                    for scheduler in schedulers:
                        scheduler.step()

                if step % interval == 0:
                    tqdm_.set_description(
                        "epoch {:d} , step {:d} , loss: {:.4f}".format(epoch+1, step, epoch_loss / step))
                    if writer:
                        for name, param in model.named_parameters():
                            writer.add_histogram(name, param, step)

                        writer.add_scalar("data_loss",
                                        total_loss/total_steps)

                if step % save_step == 0 and step > 0 and self.rank == 0:
                    print("\n")
                    with torch.no_grad():
                        result = self.evaluate(model, loaders[1], log=False)
                        result["epoch"] = epoch+1
                        result["step"] = step

                        logger.info("current result of {} is {}".format(self.name, result))
                        if result["auc"] > best_res["auc"]:
                            best_res = result
                            self.save(model, epoch+1, step, optimizers)
                            self._log(result)

                        # elif result["auc"] - best_res["auc"] < -0.05:
                        #     logger.info("model is overfitting, the result is {}, force shutdown".format(result))
                        #     return best_res

                # if self.world_size > 1:
                #     print('fuck the last barrier')
                #     barrier()
                #     print('fuck the last barrier after')

                total_steps += 1

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


        logger.info("training...")
        loss_func = self._get_loss(model)
        optimizers, schedulers = self._get_optim(model, loaders[0])

        res = self._tune(model, loaders, optimizers, loss_func, schedulers=schedulers,
                        writer=writer, interval=self.interval, save_step=int(len(loaders[0])/self.val_freq)-1)

        self._log(res)


    @torch.no_grad()
    def test(self, model, loader_test):
        """ test the model on test dataset of MINDlarge

        Args:
            model
            loader_test: DataLoader of MINDlarge_test
        """
        self.load(model, self.epochs, self.step[0])

        logger.info("testing...")
        model.cdd_size = 1
        model.eval()

        os.makedirs("data/results/{}".format(self.name), exist_ok=True)

        save_path = save_directory + "/{}_epoch{}_step{}_[k={}].txt".format(
            self.scale, self.epochs, self.step[0], self.k)

        with open(save_path, "w") as f:
            preds = []
            imp_indexes = []
            for x in tqdm(loader_test, smoothing=0):
                preds.extend(model(x).tolist())
                imp_indexes.extend(x["impression_index"])

            all_keys = list(set(imp_indexes))
            all_keys.sort()
            group_preds = {k: [] for k in all_keys}

            for i, p in zip(imp_indexes, preds):
                group_preds[i].append(p)

            for k, v in group_preds.items():
                array = np.asarray(v)
                rank_list = ss.rankdata(1 - array, method="ordinal")
                line = str(k) + " [" + ",".join([str(i)
                                                for i in rank_list]) + "]" + "\n"
                f.write(line)

        logger.info("written to prediction at {}!".format(save_path))



    @torch.no_grad()
    # FIXME: model.batch_size cannot be reached from DDP
    def encode(self, model, loader):
        """
            Encode news of self.scale in each mode, currently force to encode dev dataset
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

        encoder = model.encoder

        news_reprs = torch.zeros((news_num + 1, encoder.hidden_dim))
        news_embeddings = torch.zeros(
            (news_num + 1, encoder.signal_length, encoder.level, encoder.hidden_dim))

        start_pos = 0
        for x in tqdm(loader):
            if x['cdd_encoded_index'].shape[0] != model.batch_size:
                model.batch_size = x['cdd_encoded_index'].shape[0]

            news = x['cdd_encoded_index'].long()
            news_embedding, news_repr = model.encoder(
                news,
                news_id=x['cdd_id'].long(),
                attn_mask=x['cdd_encoded_index_pad'])

            news_reprs[start_pos:start_pos+model.batch_size] = news_repr
            news_embeddings[start_pos:start_pos+model.batch_size] = news_embedding

            start_pos += model.batch_size

        torch.save(news_reprs, "data/tensors/news_repr_{}_{}-[{}].tensor".format(
            scale, mode, self.name))
        torch.save(news_embeddings, "data/tensors/news_embedding_{}_{}-[{}].tensor".format(
            scale, mode, self.name))
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