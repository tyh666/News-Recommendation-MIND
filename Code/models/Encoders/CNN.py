import torch
import torch.nn as nn
from ..Attention import Attention

class CNN_Encoder(nn.Module):
    """ encode news through 1-d CNN and combine embeddings with personalized attention

    Args:
        news_batch: tensor of [batch_size, *, signal_length]

    Returns:
        news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, level, hidden_dim]
        news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
    """
    def __init__(self, config, vocab):
        super().__init__()
        self.name = 'cnn'

        self.dropout_p = config.dropout_p

        self.level = 1
        self.hidden_dim = config.filter_num
        self.embedding_dim = config.embedding_dim

        # pretrained embedding
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors,sparse=config.spadam,freeze=False)

        self.wordQueryProject = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.CNN = nn.Conv1d(in_channels=self.embedding_dim,
                             out_channels=self.hidden_dim, kernel_size=3, padding=1)
        self.LayerNorm = nn.LayerNorm(self.hidden_dim)

        self.query_words = nn.Parameter(torch.randn(
            (1, self.hidden_dim), requires_grad=True))

        self.RELU = nn.ReLU()
        self.DropOut = nn.Dropout(p=config.dropout_p)
        self.Tanh = nn.Tanh()


    def forward(self, news_batch, **kwargs):
        news_embedding_pretrained = self.DropOut(self.embedding(
            news_batch)).view(-1, news_batch.shape[-1], self.embedding_dim).transpose(-2, -1)
        news_embedding = self.RELU(self.LayerNorm(self.CNN(
            news_embedding_pretrained).transpose(-2, -1))).view(news_batch.shape + (self.hidden_dim,))

        news_repr = Attention.ScaledDpAttention(self.query_words, self.Tanh(self.wordQueryProject(news_embedding)), news_embedding).squeeze(dim=-2)
        return news_embedding.unsqueeze(dim=-2), news_repr
