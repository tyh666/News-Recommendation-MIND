import torch
import torch.nn as nn
from ..Modules.Attention import scaled_dp_attention, MultiheadAttention, get_attn_mask

class MHA_Encoder(nn.Module):
    def __init__(self, manager):
        super().__init__()
        self.hidden_dim = manager.hidden_dim
        self.embedding_dim = manager.bert_dim
        self.head_num = manager.head_num

        value_dim, x = divmod(self.hidden_dim, self.head_num)
        assert x == 0, "hidden_dim {} must divide head_num {}".format(self.hidden_dim, self.head_num)

        self.mha = MultiheadAttention(self.embedding_dim, self.head_num, value_dim=value_dim)
        self.query_words = nn.Parameter(torch.randn(1, self.hidden_dim))

        self.layerNorm = nn.LayerNorm(self.hidden_dim)
        self.dropOut = nn.Dropout(p=manager.dropout_p)

    def forward(self, news_embedding, attn_mask=None):
        """ encode news through multi-head self attention

        Args:
            news_embedding: tensor of [batch_size, *, signal_length, embedding_dim]
            attn_mask: tensor of [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        batch_size = news_embedding.size(0)
        signal_length = news_embedding.size(-2)
        extended_attn_mask = get_attn_mask(attn_mask.view(-1, signal_length))

        encoded_embedding = self.mha(news_embedding.view(-1, signal_length, self.embedding_dim), extended_attn_mask)
        encoded_embedding = self.dropOut(self.layerNorm(encoded_embedding)).view(batch_size, -1, signal_length, self.hidden_dim)
        news_repr = scaled_dp_attention(self.query_words, encoded_embedding, encoded_embedding, attn_mask.view(batch_size, -1, 1, signal_length)).squeeze(-2)
        return encoded_embedding, news_repr


class MHA_User_Encoder(nn.Module):
    def __init__(self, manager):
        super().__init__()
        self.name = 'mha-u'

        self.hidden_dim = manager.hidden_dim

        head_num = manager.head_num
        value_dim, x = divmod(self.hidden_dim, head_num)
        assert x == 0, "hidden_dim {} must divide head_num {}".format(self.hidden_dim, head_num)
        self.mha = MultiheadAttention(self.hidden_dim, manager.head_num, value_dim=value_dim)

        self.query_news = nn.Parameter(torch.randn(1, self.hidden_dim))
        self.layerNorm = nn.LayerNorm(self.hidden_dim)
        self.dropOut = nn.Dropout(p=manager.dropout_p)

    def forward(self, news_repr, his_mask=None, **kargs):
        """
        encode user history into a representation vector

        Args:
            news_repr: batch of news representations, [batch_size, *, hidden_dim]

        Returns:
            user_repr: user representation (coarse), [batch_size, 1, hidden_dim]
        """
        if his_mask is not None:
            extended_attn_mask = get_attn_mask(his_mask.squeeze(-1))
            news_repr = self.mha(news_repr, extended_attn_mask)
            user_repr = scaled_dp_attention(self.query_news, news_repr, news_repr, his_mask)
        else:
            news_repr = self.mha(news_repr)
            user_repr = scaled_dp_attention(self.query_news, news_repr, news_repr)
        return user_repr