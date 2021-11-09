# Two tower baseline
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from .TwoTowerBaseModel import TwoTowerBaseModel
from .Encoders.BERT import BERT_Encoder
from models.UniLM.configuration_tnlrv3 import TuringNLRv3Config
from models.UniLM.modeling import TuringNLRv3ForSequenceClassification, relative_position_bucket

class PLM(TwoTowerBaseModel):
    """
    Tow tower model with selection

    1. encode candidate news with bert
    2. encode ps terms with the same bert, using [CLS] embedding as user representation
    3. predict by scaled dot product
    """
    def __init__(self, manager, encoderU):
        super().__init__(manager)

        self.encoderU = encoderU

        if manager.debias:
            self.userBias = nn.Parameter(torch.randn(1,manager.bert_dim))
            nn.init.xavier_normal_(self.userBias)

        if manager.bert == 'unilm':
            config = TuringNLRv3Config.from_pretrained(manager.unilm_config_path)
            bert = TuringNLRv3ForSequenceClassification.from_pretrained(manager.unilm_path, config=config).bert

        elif manager.bert == 'deberta':
            # add a pooler
            bert = AutoModel.from_pretrained(
                manager.get_bert_for_load(),
                cache_dir=manager.path + 'bert_cache/'
            )
            self.pooler = nn.Sequential(
                nn.Linear(manager.bert_dim, manager.bert_dim),
                nn.GELU()
            )

        else:
            bert = AutoModel.from_pretrained(
                manager.get_bert_for_load(),
                cache_dir=manager.path + 'bert_cache/'
            )

        self.bert = bert

        manager.name = '__'.join(["plm", manager.bert, manager.encoderU, manager.granularity])
        # used in fast evaluate
        self.name = manager.name


    def encode_news(self, x):
        """
        encode news of loader_news
        """
        # encode news with MIND_news
        batch_size = x['cdd_encoded_index'].size(0)
        if self.granularity != 'token':
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

        cdd_news = x["cdd_encoded_index"].to(self.device).view(-1, self.signal_length)
        cdd_attn_mask = cdd_attn_mask.view(-1, self.signal_length)
        cdd_news_repr = self.bert(cdd_news, cdd_attn_mask)[-1]
        if hasattr(self, 'pooler'):
            cdd_news_repr = self.pooler(cdd_news_repr[:, 0])
        cdd_news_repr = cdd_news_repr.view(batch_size, -1, self.hidden_dim)
        return cdd_news_repr


    def encode_user(self, x):
        # fast encoding user
        if hasattr(self, "news_reprs"):
            his_news_repr = self.news_reprs(x['his_id'].to(self.device))

        # slow encoding user, PLM per historical piece
        else:
            batch_size = x['his_encoded_index'].size(0)

            if self.granularity != 'token':
                batch_size = x['his_encoded_index'].size(0)
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
            else:
                his_subword_prefix = None
                his_attn_mask = x["his_attn_mask"].to(self.device)

            his_news = x["his_encoded_index"].to(self.device).view(-1, self.signal_length)
            his_attn_mask = his_attn_mask.view(-1, self.signal_length)
            his_news_repr = self.bert(his_news, his_attn_mask)[-1]
            if hasattr(self, 'pooler'):
                his_news_repr = self.pooler(his_news_repr[:, 0])
            his_news_repr = his_news_repr.view(batch_size, self.his_size, self.hidden_dim)

        user_repr = self.encoderU(his_news_repr)
        if hasattr(self, 'userBias'):
            user_repr = user_repr + self.userBias

        return user_repr, None