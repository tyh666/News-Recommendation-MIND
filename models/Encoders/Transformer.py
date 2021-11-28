import torch
import torch.nn as nn
from models.Modules.Attention import scaled_dp_attention
from ..Modules.OneLayerBert import BertLayer
from transformers import BertConfig


class Transformer_Encoder(nn.Module):
    def __init__(self, manager):
        super().__init__()
        bert_config = BertConfig()
        bert_config.all_head_size = manager.hidden_dim
        self.hidden_dim = manager.hidden_dim

        self.project = nn.Linear(bert_config.hidden_size, bert_config.all_head_size)
        self.bert = BertLayer(bert_config)
        self.query_words = nn.Parameter(torch.randn((1, self.hidden_dim)))
        self.Tanh = nn.Tanh()
        nn.init.xavier_normal_(self.query_words)


    def forward(self, news_embedding, attn_mask=None):
        """ encode news by one-layer bert

        Args:
            news_embedding: tensor of [batch_size, *, signal_length, embedding_dim]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        batch_size, news_num, signal_length, embedding_dim = news_embedding.shape

        bert_input = self.project(news_embedding.view(-1, signal_length, embedding_dim))
        attn_mask = attn_mask.view(-1, 1, signal_length)

        bert_output = self.bert(bert_input, attn_mask)
        news_repr = scaled_dp_attention(self.query_words, bert_output, bert_output, attn_mask=attn_mask).view(batch_size, -1, self.hidden_dim)
        return bert_output.view(batch_size, news_num, signal_length, self.hidden_dim), news_repr
