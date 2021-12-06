import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .TwoTowerBaseModel import TwoTowerBaseModel
from .Encoders.BERT import BERT_Encoder

class TESRec(TwoTowerBaseModel):
    """
    Tow tower model with selection, one tower user modeling

    1. encode candidate news with bert
    2. encode ps terms with the same bert
    3. predict by scaled dot product
    """
    def __init__(self, manager, embedding, encoderN, encoderU, reducer, aggregator=None):
        super().__init__(manager)

        self.embedding = embedding
        self.reducer = reducer
        self.aggregator = aggregator
        self.bert = BERT_Encoder(manager)

        self.reducer_name = manager.reducer

        # personalized keywords selection
        if manager.reducer == "personalized":
            self.encoderN = encoderN
            self.encoderU = encoderU
        # global keywords selection
        elif manager.reducer == "global":
            self.encoderN = encoderN
            self.query = nn.Parameter(torch.randn(1, manager.bert_dim))
            self.queryProject = nn.Linear(manager.bert_dim, encoderN.hidden_dim)
            nn.init.xavier_uniform_(self.query)

        if manager.debias:
            self.userBias = nn.Parameter(torch.randn(1,self.bert.hidden_dim))
            nn.init.xavier_normal_(self.userBias)

        self.hidden_dim = manager.bert_dim

        if aggregator is not None:
            manager.name = '__'.join(['tesrec', manager.bert, manager.encoderN, manager.encoderU, manager.reducer, manager.aggregator, "token", str(manager.k)])
        else:
            manager.name = '__'.join(['tesrec', manager.bert, manager.encoderN, manager.encoderU, manager.reducer, str(manager.k), manager.verbose])
        # used in fast evaluate
        self.name = manager.name


    def encode_news(self, x):
        """
        encode candidate news
        """
        # encode news with MIND_news
        cdd_news = x["cdd_encoded_index"].to(self.device)
        cdd_attn_mask = x['cdd_attn_mask'].to(self.device)

        _, cdd_news_repr, _ = self.bert(
            self.embedding(cdd_news), cdd_attn_mask
        )

        return cdd_news_repr


    def encode_user(self, x):
        """
        encoder user
        """
        batch_size = x['his_encoded_index'].size(0)
        his_news = x["his_encoded_index"].to(self.device)
        his_attn_mask = x["his_attn_mask"].to(self.device)
        if 'his_refined_mask' in x:
            his_refined_mask = x["his_refined_mask"].to(self.device)

        his_news_embedding = self.embedding(his_news)
        if hasattr(self, 'encoderN'):
            his_news_encoded_embedding,  his_news_repr = self.encoderN(
                his_news_embedding, his_attn_mask
            )
        else:
            his_news_encoded_embedding = None
            his_news_repr = None
        # no need to calculate this if ps_terms are fixed in advance

        if self.reducer_name == 'personalized':
            user_repr_ext = self.encoderU(his_news_repr, his_mask=x['his_mask'], user_index=x['user_id'].to(self.device))
        elif self.reducer_name == "global":
            user_repr_ext = self.queryProject(self.query).expand(batch_size, 1, -1)
        else:
            user_repr_ext = None

        # print(user_repr_ext)

        ps_terms, ps_term_mask, kid = self.reducer(his_news_encoded_embedding, his_news_embedding, user_repr_ext, his_news_repr, his_attn_mask, his_refined_mask)

        _, user_repr, _ = self.bert(ps_terms, ps_term_mask, ps_term_input=True)

        if self.aggregator is not None:
            user_repr = self.aggregator(user_repr)

        if hasattr(self, 'userBias'):
            user_repr = user_repr + self.userBias

        return user_repr, kid