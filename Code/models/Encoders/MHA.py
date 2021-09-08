import torch
import torch.nn as nn
from ..Modules.Attention import ScaledDpAttention, MultiheadAttention

class MHA_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = 'mha'

        self.hidden_dim = config.hidden_dim
        self.embedding_dim = config.embedding_dim
        self.head_num = config.head_num

        value_dim, x = divmod(self.hidden_dim, self.head_num)
        assert x == 0, "hidden_dim {} must divide head_num {}".format(self.hidden_dim, self.head_num)

        self.mha = MultiheadAttention(self.embedding_dim, self.head_num, value_dim=value_dim)
        self.query_words = nn.Parameter(torch.randn(1, self.hidden_dim))

        self.softMax = nn.Softmax(dim=-1)
        self.layerNorm = nn.LayerNorm(self.hidden_dim)
        self.dropOut = nn.Dropout(p=config.dropout_p)

    def forward(self, news_embedding):
        """ encode news through multi-head self attention

        Args:
            news_embedding: tensor of [batch_size, *, signal_length, embedding_dim]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        batch_size = news_embedding.size(0)
        signal_length = news_embedding.size(-2)
        encoded_embedding = self.mha(news_embedding.view(-1, signal_length, self.embedding_dim))
        encoded_embedding = self.dropOut(self.layerNorm(encoded_embedding)).view(batch_size, -1, signal_length, self.hidden_dim)
        news_repr = ScaledDpAttention(self.query_words, encoded_embedding, encoded_embedding).squeeze(-2)
        return encoded_embedding, news_repr


class MHA_User_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = 'mha-u'

        self.hidden_dim = config.hidden_dim

        head_num = config.head_num
        value_dim, x = divmod(self.hidden_dim, head_num)
        assert x == 0, "hidden_dim {} must divide head_num {}".format(self.hidden_dim, head_num)
        self.mha = MultiheadAttention(self.hidden_dim, config.head_num, value_dim=value_dim)

        self.query_news = nn.Parameter(torch.randn(1, self.hidden_dim))
        self.tanh = nn.Tanh()

    def forward(self, news_reprs):
        """
        encode user history into a representation vector

        Args:
            news_reprs: batch of news representations, [batch_size, *, hidden_dim]

        Returns:
            user_repr: user representation (coarse), [batch_size, 1, hidden_dim]
        """
        news_reprs = self.mha(news_reprs)
        user_repr = ScaledDpAttention(self.query_news, news_reprs, news_reprs)
        return user_repr