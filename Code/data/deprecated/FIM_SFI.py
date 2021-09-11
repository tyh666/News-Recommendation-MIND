import torch
import torch.nn as nn
from ..Modules.Attention import Attention

class FIM_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = 'fim'

        self.kernel_size = 3

        self.level = 3
        config.level = self.level

        self.hidden_dim = config.hidden_dim
        self.embedding_dim = config.embedding_dim

        self.ReLU = nn.ReLU()
        self.LayerNorm = nn.LayerNorm(self.hidden_dim)
        self.DropOut = nn.Dropout(p=config.dropout_p)

        self.query_words = nn.Parameter(torch.randn(
            (1, self.hidden_dim), requires_grad=True))
        self.query_levels = nn.Parameter(torch.randn(
            (1, self.hidden_dim), requires_grad=True))

        self.CNN_d1 = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size, dilation=1, padding=1)
        self.CNN_d2 = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size, dilation=2, padding=2)
        self.CNN_d3 = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size, dilation=3, padding=3)

        nn.init.xavier_normal_(self.CNN_d1.weight)
        nn.init.xavier_normal_(self.CNN_d2.weight)
        nn.init.xavier_normal_(self.CNN_d3.weight)
        nn.init.xavier_normal_(self.query_levels)
        nn.init.xavier_normal_(self.query_words)

    def _HDC(self, news_embedding_set):
        """ stack 1d CNN with dilation rate expanding from 1 to 3

        Args:
            news_embedding_set: tensor of [set_size, signal_length, embedding_dim]

        Returns:
            news_embedding_dilations: tensor of [set_size, signal_length, levels(3), filter_num]
        """

        # don't know what d_0 meant in the original paper
        news_embedding_dilations = torch.zeros(
            (news_embedding_set.shape[0], news_embedding_set.shape[1], self.level, self.hidden_dim), device=news_embedding_set.device)

        news_embedding_set = news_embedding_set.transpose(-2,-1)

        news_embedding_d1 = self.CNN_d1(news_embedding_set)
        news_embedding_d1 = self.LayerNorm(news_embedding_d1.transpose(-2,-1))
        news_embedding_dilations[:,:,0,:] = self.ReLU(news_embedding_d1)

        news_embedding_d2 = self.CNN_d2(news_embedding_set)
        news_embedding_d2 = self.LayerNorm(news_embedding_d2.transpose(-2,-1))
        news_embedding_dilations[:,:,1,:] = self.ReLU(news_embedding_d2)

        news_embedding_d3 = self.CNN_d3(news_embedding_set)
        news_embedding_d3 = self.LayerNorm(news_embedding_d3.transpose(-2,-1))
        news_embedding_dilations[:,:,2,:] = self.ReLU(news_embedding_d3)

        return news_embedding_dilations

    def forward(self, news_embedding):
        """ encode news through stacked dilated CNN

        Args:
            news_embedding: tensor of [batch_size, *, signal_length, embedding_dim]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, level, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        batch_size = news_embedding.size(0)
        news_num = news_embedding.size(1)
        signal_length = news_embedding.size(2)
        news_embedding = news_embedding.view(-1, signal_length, self.embedding_dim)
        news_embedding = self._HDC(news_embedding).view(batch_size, news_num, signal_length, self.level, self.hidden_dim)
        news_embedding_attn = Attention.scaled_dp_attention(
            self.query_levels, news_embedding, news_embedding).squeeze(dim=-2)
        news_repr = Attention.scaled_dp_attention(self.query_words, news_embedding_attn, news_embedding_attn).squeeze(
            dim=-2).view(batch_size, news_num, self.hidden_dim)

        return news_embedding, news_repr


if __name__ == '__main__':
    from models.Encoders.FIM import FIM_Encoder
    from data.configs.demo import config
    config.embedding_dim = 5
    config.hidden_dim = 6
    a = torch.rand(2,3,4,5)

    enc = FIM_Encoder(config)
    b = enc(a)
    print(b[0].size(),b[1].size())