import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from .TwoTowerBaseModel import TwoTowerBaseModel

class XFormer(TwoTowerBaseModel):
    def __init__(self, manager):
        """
        one tower user modeling baseline
        """
        super().__init__(manager)

        self.bert_name = manager.bert
        self.max_length, self.max_length_per_history = manager.get_max_length_for_truncating()

        if self.bert_name == "reformer":
            from transformers import ReformerModel, ReformerConfig
            config = ReformerConfig()
            config.axial_pos_shape = (40,32)
            self.reformer = ReformerModel(config)

            self.bert = AutoModel.from_pretrained(
                "bert-base-uncased",
                cache_dir=manager.path + 'bert_cache/'
            )

            self.pooler = nn.Sequential(
                nn.Dropout(p=0.05),
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Tanh(),
                nn.Dropout(p=0.05),
                nn.Linear(config.hidden_size, config.hidden_size)
            )

            self.projector = nn.Linear(self.bert.config.hidden_size, self.reformer.config.hidden_size)
            # self.bert.config.max_embedding_size = self.max_length_per_history * self.his_size + 1
            self.register_buffer("pad_token_id", 2 * torch.ones(1, 1, 1, dtype=torch.long), persistent=False)
            self.register_buffer("pad_token_mask", torch.zeros(1, 1, 1, dtype=torch.long), persistent=False)

            self.hidden_dim = config.hidden_size

        else:
            self.bert = AutoModel.from_pretrained(
                manager.get_bert_for_load(),
                cache_dir=manager.path + 'bert_cache/'
            )

        if manager.debias:
            self.userBias = nn.Parameter(torch.randn(1,manager.bert_dim))
            nn.init.xavier_normal_(self.userBias)

        manager.name = '__'.join(['xformer', manager.bert])
        # used in fast evaluate
        self.name = manager.name


    def encode_news(self, x):
        """
        encode news of loader_news
        """
        # encode news with MIND_news
        batch_size = x['cdd_encoded_index'].size(0)
        cdd_news = x["cdd_encoded_index"].to(self.device).view(-1, self.signal_length)
        cdd_attn_mask = x['cdd_attn_mask'].to(self.device).view(-1, self.signal_length)

        bert_output = self.bert(cdd_news, cdd_attn_mask)

        if self.bert_name in ["reformer"]:
            cdd_news_repr = self.projector(bert_output.pooler_output).view(batch_size, -1, self.hidden_dim)
        else:
            cdd_news_repr = bert_output.pooler_output.view(batch_size, -1, self.hidden_dim)

        return cdd_news_repr


    def encode_user(self,x):
        batch_size = x['his_encoded_index'].size(0)
        his_news = x["his_encoded_index"].to(self.device)
        his_attn_mask = x["his_attn_mask"].to(self.device)

        cls_token_id = his_news[:, 0, [0]]
        his_news = his_news[:, :, 1:self.max_length_per_history + 1].reshape(batch_size, -1)[:, :self.max_length - 1]
        his_news = torch.cat([cls_token_id, his_news], dim=-1)

        cls_token_mask = his_attn_mask[:, 0, [0]]
        his_attn_mask = his_attn_mask[:, :, 1:self.max_length_per_history + 1].reshape(batch_size, -1)[:, :self.max_length - 1]
        his_attn_mask = torch.cat([cls_token_mask, his_attn_mask], dim=-1)

        if self.bert_name in ["reformer"]:
            user_repr = self.pooler(self.reformer(his_news, his_attn_mask)[0][:, 0])
        else:
            user_repr = self.bert(his_news, his_attn_mask).pooler_output
        user_repr = user_repr.unsqueeze(1)

        if hasattr(self, 'userBias'):
            user_repr = user_repr + self.userBias

        return user_repr, None
