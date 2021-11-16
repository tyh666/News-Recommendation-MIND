# Two tower baseline
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Modules.Attention import scaled_dp_attention
from .Encoders.CNN import CNN_Encoder
from .Encoders.Pooling import Attention_Pooling
from .TwoTowerBaseModel import TwoTowerBaseModel

class NAML(TwoTowerBaseModel):
    def __init__(self, manager):
        super().__init__(manager)

        # only used for test
        self.embedding = nn.Embedding(30522, 300)
        self.encoderN = CNN_Encoder(manager)
        self.encoderU = Attention_Pooling(manager)

        self.query = nn.Parameter(torch.rand((1, manager.hidden_dim)))
        manager.name = "lstur"
        # used in fast evaluate
        self.name = manager.name


    def encode_user(self, x):
        his_news = x["his_encoded_index"].to(self.device)
        his_news_embedding = self.embedding(his_news)

        his_title = his_news_embedding[:, :, :30]
        his_abs = his_news_embedding[:, :, 30:]

        _, his_title_repr = self.encoderN(his_title)
        _, his_abs_repr = self.encoderN(his_abs)

        user_repr_1 = self.encoderU(his_title_repr).unsqueeze(-2)
        user_repr_2 = self.encoderU(his_abs_repr).unsqueeze(-2)

        user_reprs = torch.cat([user_repr_1, user_repr_2], dim=-2)
        user_repr = scaled_dp_attention(self.query, user_reprs, user_reprs)
        return user_repr, None