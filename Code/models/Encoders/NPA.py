import torch.nn as nn
from ..Attention import Attention


class NPA_Encoder(nn.Module):
    def __init__(self, config, vocab, user_num):
        super().__init__()
        self.name = 'npa-encoder'

        self.dropout_p = config.dropout_p

        self.level = 1
        self.hidden_dim = config.filter_num
        self.embedding_dim = config.embedding_dim
        self.user_dim = config.user_dim
        self.query_dim = config.query_dim

        # pretrained embedding
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors,sparse=config.spadam,freeze=False)

        # trainable lookup layer for user embedding, important to have len(uid2idx) + 1 rows because user indexes start from 1
        self.user_embedding = nn.Embedding(user_num + 1, self.user_dim, sparse=True)
        self.user_embedding.weight.requires_grad = True

        # project e_u to word query preference vector of query_dim
        self.wordPrefProject = nn.Linear(self.user_dim, self.query_dim)
        # project preference query to vector of hidden_dim
        self.wordQueryProject = nn.Linear(self.query_dim, self.hidden_dim)

        # input tensor shape is [batch_size,in_channels,signal_length]
        # in_channels is the length of embedding, out_channels indicates the number of filters, signal_length is the length of title
        # set paddings=1 to get the same length of title, referring M in the paper
        self.CNN = nn.Conv1d(in_channels=self.embedding_dim,
                             out_channels=self.hidden_dim, kernel_size=3, padding=1)
        self.RELU = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.DropOut = nn.Dropout(p=config.dropout_p)


    def forward(self, news_batch, **kwargs):
        """ encode news through 1-d CNN and combine embeddings with personalized attention

        Args:
            news_batch: tensor of [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        e_u = self.DropOut(self.user_embedding(kwargs['user_index']))
        word_query = self.Tanh(self.wordQueryProject(
            self.RELU(self.wordPrefProject(e_u))))

        news_embedding_pretrained = self.DropOut(self.embedding(
            news_batch)).view(-1, news_batch.shape[-1], self.embedding_dim).transpose(-2, -1)
        news_embedding = self.RELU(self.CNN(
            news_embedding_pretrained)).transpose(-2, -1).view(news_batch.shape + (self.hidden_dim,))

        news_repr = Attention.ScaledDpAttention(word_query.view(
            word_query.shape[0], 1, 1, word_query.shape[-1]), news_embedding, news_embedding).squeeze(dim=-2)
        return news_embedding.view(news_batch.shape + (self.level, self.hidden_dim)), news_repr
