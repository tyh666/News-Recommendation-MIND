# Two tower baseline
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .TwoTowerBaseModel import TwoTowerBaseModel
from .Encoders.BERT import BERT_Encoder

class TESRec(TwoTowerBaseModel):
    """
    Tow tower model with selection

    1. encode candidate news with bert
    2. encode ps terms with the same bert, using [CLS] embedding as user representation
    3. predict by scaled dot product
    """
    def __init__(self, manager, embedding, encoderN, encoderU, reducer, aggregator=None):
        super().__init__(manager)

        self.embedding = embedding
        # only these reducers need selection encoding
        if manager.reducer in manager.get_need_encode_reducers():
            self.encoderN = encoderN
            # self.encoderU = encoderU
        self.reducer = reducer
        self.aggregator = aggregator
        self.bert = BERT_Encoder(manager)
        self.query = nn.Parameter(torch.randn(1, manager.bert_dim))

        self.queryProject = nn.Linear(manager.bert_dim, encoderN.hidden_dim)

        nn.init.xavier_uniform_(self.query)

        if manager.debias:
            self.userBias = nn.Parameter(torch.randn(1,self.bert.hidden_dim))
            nn.init.xavier_normal_(self.userBias)

        self.hidden_dim = manager.bert_dim

        if aggregator is not None:
            manager.name = '__'.join(['tesrec', manager.bert, manager.encoderN, manager.encoderU, manager.reducer, manager.aggregator, manager.granularity, str(manager.k)])
        else:
            manager.name = '__'.join(['tesrec', manager.bert, manager.encoderN, manager.encoderU, manager.reducer, manager.granularity, str(manager.k), manager.verbose])
        # used in fast evaluate
        self.name = manager.name


    def encode_news(self, x):
        """
        encode candidate news
        """
        # encode news with MIND_news
        if self.granularity != 'token':
            batch_size = x['cdd_subword_index'].size(0)
            cdd_dest = self.cdd_dest[:batch_size]
            cdd_subword_index = x['cdd_subword_index'].to(self.device)
            cdd_subword_index = cdd_subword_index[:, :, 0] * self.signal_length + cdd_subword_index[:, :, 1]

            cdd_subword_prefix = cdd_dest.scatter(dim=-1, index=cdd_subword_index, value=1)
            cdd_subword_prefix = cdd_subword_prefix.view(batch_size, self.signal_length, self.signal_length)

            if self.granularity == 'avg':
                # average subword embeddings as the word embedding
                cdd_subword_prefix = F.normalize(cdd_subword_prefix, p=1, dim=-1)
            cdd_attn_mask = cdd_subword_prefix.matmul(x['cdd_attn_mask'].to(self.device).float().unsqueeze(-1)).squeeze(-1)

        else:
            cdd_subword_prefix = None
            cdd_attn_mask = x['cdd_attn_mask'].to(self.device)

        cdd_news = x["cdd_encoded_index"].to(self.device)
        _, cdd_news_repr, _ = self.bert(
            self.embedding(cdd_news, cdd_subword_prefix), cdd_attn_mask
        )

        return cdd_news_repr


    def encode_user(self, x):
        """
        encoder user
        """
        batch_size = x['his_encoded_index'].size(0)
        if self.granularity != 'token':
            his_dest = self.his_dest[:batch_size]

            his_subword_index = x['his_subword_index'].to(self.device)
            his_signal_length = his_subword_index.size(-2)
            his_subword_index = his_subword_index[:, :, :, 0] * his_signal_length + his_subword_index[:, :, :, 1]

            his_subword_prefix = his_dest.scatter(dim=-1, index=his_subword_index, value=1) * x["his_mask"].to(self.device)
            his_subword_prefix = his_subword_prefix.view(batch_size, self.his_size, his_signal_length, his_signal_length)

            if self.granularity == 'avg':
                # average subword embeddings as the word embedding
                his_subword_prefix = F.normalize(his_subword_prefix, p=1, dim=-1)

            his_attn_mask = his_subword_prefix.matmul(x["his_attn_mask"].to(self.device).float().unsqueeze(-1)).squeeze(-1)
            his_refined_mask = None
            if 'his_refined_mask' in x:
                his_refined_mask = his_subword_prefix.matmul(x["his_refined_mask"].to(self.device).float().unsqueeze(-1)).squeeze(-1)

        else:
            his_subword_prefix = None
            his_attn_mask = x["his_attn_mask"].to(self.device)
            his_refined_mask = None
            if 'his_refined_mask' in x:
                his_refined_mask = x["his_refined_mask"].to(self.device)

        his_news = x["his_encoded_index"].to(self.device)
        his_news_embedding = self.embedding(his_news, his_subword_prefix)
        if hasattr(self, 'encoderN'):
            his_news_encoded_embedding, his_news_repr = self.encoderN(
                his_news_embedding, his_attn_mask
            )
        else:
            his_news_encoded_embedding = None
            his_news_repr = None
        # no need to calculate this if ps_terms are fixed in advance

        if self.reducer.name == 'matching':
            user_repr_ext = self.queryProject(self.query).expand(batch_size, 1, -1)
            # user_repr_ext = self.encoderU(his_news_repr, his_mask=x['his_mask'], user_index=x['user_id'].to(self.device))
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