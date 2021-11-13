# Two tower baseline
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Embeddings.BERT import BERT_Embedding
from .Encoders.MHA import MHA_Encoder, MHA_User_Encoder
from .TwoTowerBaseModel import TwoTowerBaseModel

class NRMS(TwoTowerBaseModel):
    def __init__(self, manager):
        super().__init__(manager)

        self.embedding = BERT_Embedding(manager)
        self.encoderN = MHA_Encoder(manager)
        self.encoderU = MHA_User_Encoder(manager)

        manager.name = "nrms"
        # used in fast evaluate
        self.name = manager.name


    def encode_news(self, x):
        """
        encode news of loader_news
        """
        # encode news with MIND_news
        cdd_news = x["cdd_encoded_index"].to(self.device)
        cdd_news_embedding = self.embedding(cdd_news)
        cdd_attn_mask = x["cdd_attn_mask"].to(self.device)

        _, cdd_news_repr = self.encoderN(cdd_news_embedding, cdd_attn_mask)
        return cdd_news_repr


    def encode_user(self, x):
        his_news = x["his_encoded_index"].to(self.device)
        his_news_embedding = self.embedding(his_news)
        his_attn_mask = x["his_attn_mask"].to(self.device)

        _, his_news_repr = self.encoderN(his_news_embedding, his_attn_mask)
        user_repr = self.encoderU(his_news_repr, his_mask=x["his_mask"].to(self.device))
        return user_repr, None