import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from .TwoTowerBaseModel import TwoTowerBaseModel

class XFormer(TwoTowerBaseModel):
    def __init__(self, manager):
        super().__init__(manager)

        self.bert = AutoModel.from_pretrained(
            manager.get_bert_for_load(),
            cache_dir=manager.path + 'bert_cache/'
        )

        self.bert_name = manager.bert
        self.max_length, self.max_length_per_history = manager.get_max_length_for_truncating()

        if self.bert_name == "reformer":
            self.pooler = nn.Sequential(
                nn.Dropout(p=0.05),
                nn.Linear(self.bert.config.hidden_size * 2, self.bert.config.hidden_size),
                nn.Tanh(),
                nn.Dropout(p=0.05),
                nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
            )
            # self.bert.config.max_embedding_size = self.max_length_per_history * self.his_size + 1
            self.register_buffer("pad_token_id", 2 * torch.ones(1, 1, 1), persistent=False)
            self.register_buffer("pad_token_mask", torch.zeros(1, 1, 1))
            self.bert.config.axial_pos_shape = (40, 64)

        if manager.debias:
            self.userBias = nn.Parameter(torch.randn(1,manager.bert_dim))
            nn.init.xavier_normal_(self.userBias)

        manager.name = '__'.join(['xformer', manager.bert, manager.granularity])
        # used in fast evaluate
        self.name = manager.name


    def encode_news(self, x):
        """
        encode news of loader_news
        """
        # encode news with MIND_news
        B, N, L = x['cdd_encoded_index'].size()
        if self.granularity != 'token':
            cdd_dest = self.cdd_dest[:B]
            cdd_subword_index = x['cdd_subword_index'].to(self.device)
            cdd_subword_index = cdd_subword_index[:, :, 0] * self.signal_length + cdd_subword_index[:, :, 1]

            cdd_subword_prefix = cdd_dest.scatter(dim=-1, index=cdd_subword_index, value=1)
            cdd_subword_prefix = cdd_subword_prefix.view(B, self.signal_length, self.signal_length)

            if self.granularity == 'avg':
                # average subword embeddings as the word embedding
                cdd_subword_prefix = F.normalize(cdd_subword_prefix, p=1, dim=-1)
            cdd_attn_mask = cdd_subword_prefix.matmul(x['cdd_attn_mask'].to(self.device).float().unsqueeze(-1)).squeeze(-1)

        else:
            cdd_subword_prefix = None
            cdd_attn_mask = x['cdd_attn_mask'].to(self.device)

        cdd_news = x["cdd_encoded_index"].to(self.device).view(-1, self.signal_length)
        cdd_attn_mask = cdd_attn_mask.view(-1, self.signal_length)
        if self.bert_name in ["reformer"]:
            cdd_news = torch.cat(cdd_news, self.pad_token_id.expand(B, N, self.max_length - L))
            cdd_attn_mask = torch.cat(cdd_attn_mask, self.pad_token_mask.expand(B, N, self.max_length - L))
            cdd_news_repr = self.pooler(self.bert(cdd_news, cdd_attn_mask)[:, 0])
        else:
            cdd_news_repr = self.bert(cdd_news, cdd_attn_mask).pooler_output
        cdd_news_repr = cdd_news_repr.view(B, -1, self.hidden_dim)
        return cdd_news_repr


    def encode_user(self,x):
        batch_size = x['cdd_encoded_index'].size(0)
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

        # do not add extra [SEP]
        his_news = x["his_encoded_index"].to(self.device)
        cls_token_id = his_news[:, 0, [0]]
        his_news = his_news[:, :, 1:self.max_length_per_history + 1].reshape(batch_size, -1)[:, :self.max_length - 1]
        his_news = torch.cat([cls_token_id, his_news], dim=-1)

        cls_token_mask = his_attn_mask[:, 0, [0]]
        his_attn_mask = his_attn_mask[:, :, 1:self.max_length_per_history + 1].reshape(batch_size, -1)[:, :self.max_length - 1]
        his_attn_mask = torch.cat([cls_token_mask, his_attn_mask], dim=-1)

        if self.bert_name in ["reformer"]:
            user_repr = self.pooler(self.bert(his_news, his_attn_mask)[:, 0])
        else:
            user_repr = self.bert(his_news, his_attn_mask).pooler_output
        user_repr = user_repr.unsqueeze(1)

        if hasattr(self, 'userBias'):
            user_repr = user_repr + self.userBias

        return user_repr, None