# Two tower baseline
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Encoders.CNN import CNN_Encoder
from .Encoders.RNN import LSTUR_User_Encoder
from .TwoTowerBaseModel import TwoTowerBaseModel

class LSTUR(TwoTowerBaseModel):
    def __init__(self, manager):
        super().__init__(manager)

        # only used for test
        self.embedding = nn.Embedding(30522, 300)
        self.encoderN = CNN_Encoder(manager)
        self.encoderU = LSTUR_User_Encoder(manager)

        manager.name = "lstur"
        # used in fast evaluate
        self.name = manager.name


    def encode_user(self, x):
        his_news = x["his_encoded_index"].to(self.device)
        his_news_embedding = self.embedding(his_news)
        his_attn_mask = x["his_attn_mask"].to(self.device)

        _, his_news_repr = self.encoderN(his_news_embedding, his_attn_mask)
        user_repr = self.encoderU(his_news_repr, his_mask=x["his_mask"].to(self.device), user_index=x['user_id'].to(self.device))
        return user_repr, None