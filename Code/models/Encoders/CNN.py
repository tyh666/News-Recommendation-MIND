import torch
import torch.nn as nn
from ..Attention import Attention

class CNN_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = 'cnn'

        self.hidden_dim = config.hidden_dim
        self.embedding_dim = config.embedding_dim

        self.wordQueryProject = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.CNN = nn.Conv1d(
            in_channels=self.embedding_dim,
            out_channels=self.hidden_dim,
            kernel_size=3,
            padding=1
        )
        self.layerNorm = nn.LayerNorm(self.hidden_dim)

        self.query_words = nn.Parameter(torch.randn(
            (1, self.hidden_dim), requires_grad=True))

        self.RELU = nn.ReLU()
        self.Tanh = nn.Tanh()

        nn.init.xavier_normal_(self.wordQueryProject.weight)
        nn.init.xavier_normal_(self.CNN.weight)
        nn.init.xavier_normal_(self.query_words)


    def forward(self, news_embedding):
        """ encode news through 1-d CNN

        Args:
            news_embedding: tensor of [batch_size, *, signal_length, embedding_dim]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        signal_length = news_embedding.size(2)
        cnn_input = news_embedding.view(-1, signal_length, self.embedding_dim).transpose(-2, -1)
        cnn_output = self.RELU(self.layerNorm(self.CNN(cnn_input).transpose(-2, -1))).view(*news_embedding.shape[:-1], self.hidden_dim)

        news_repr = Attention.ScaledDpAttention(self.query_words, self.Tanh(self.wordQueryProject(cnn_output)), cnn_output).squeeze(dim=-2)
        return cnn_output, news_repr

if __name__ == '__main__':
    from models.Encoders.CNN import CNN_Encoder
    from data.configs.demo import config

    config.npratio = 1
    config.batch_size = 2
    config.his_size = 2
    config.k = 3
    config.embedding = 'bert'
    config.bert = 'bert-base-uncased'
    config.signal_length = 512

    config.embedding_dim = 768
    config.hidden_dim = 768

    a = torch.rand(2,2,512,768)
    enc = CNN_Encoder(config)
    res = enc(a)
    print(res[0].shape, res[1].shape)