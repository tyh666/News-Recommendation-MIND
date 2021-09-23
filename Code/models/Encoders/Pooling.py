import torch
import torch.nn as nn
from ..Modules.Attention import scaled_dp_attention

class Attention_Pooling(nn.Module):
    def __init__(self, manager):
        super().__init__()

        self.name = 'attention-pooling'
        self.query_news = nn.Parameter(torch.randn(1, manager.hidden_dim))
        nn.init.xavier_normal_(self.query_news)

    def forward(self, news_reprs, *args, **kargs):
        """
        encode user history into a representation vector

        Args:
            news_reprs: batch of news representations, [batch_size, *, hidden_dim]

        Returns:
            user_repr: user representation (coarse), [batch_size, 1, hidden_dim]
        """
        user_repr = scaled_dp_attention(self.query_news, news_reprs, news_reprs)
        return user_repr


class Average_Pooling(nn.Module):
    def __init__(self, manager):
        super().__init__()

        self.name = 'average-pooling'

    def forward(self, news_reprs, *args, **kargs):
        """
        encode user history into a representation vector

        Args:
            news_reprs: batch of news representations, [batch_size, *, hidden_dim]

        Returns:
            user_repr: user representation (coarse), [batch_size, 1, hidden_dim]
        """
        user_repr = news_reprs.mean(dim=1, keepdim=True)
        return user_repr


# class NPA_Encoder(nn.Module):
#     def __init__(self, manager, vocab, user_num):
#         super().__init__()
#         self.name = 'npa-encoder'

#         self.dropout_p = manager.dropout_p

#         self.level = 1
#         self.hidden_dim = manager.filter_num
#         self.embedding_dim = manager.embedding_dim
#         self.user_dim = manager.user_dim
#         self.query_dim = manager.query_dim

#         # pretrained embedding
#         self.embedding = nn.Embedding.from_pretrained(vocab.vectors,sparse=manager.spadam,freeze=False)

#         # trainable lookup layer for user embedding, important to have len(uid2idx) + 1 rows because user indexes start from 1
#         self.user_embedding = nn.Embedding(user_num + 1, self.user_dim, sparse=True)
#         self.user_embedding.weight.requires_grad = True

#         # project e_u to word query preference vector of query_dim
#         self.wordPrefProject = nn.Linear(self.user_dim, self.query_dim)
#         # project preference query to vector of hidden_dim
#         self.wordQueryProject = nn.Linear(self.query_dim, self.hidden_dim)

#         # input tensor shape is [batch_size,in_channels,signal_length]
#         # in_channels is the length of embedding, out_channels indicates the number of filters, signal_length is the length of title
#         # set paddings=1 to get the same length of title, referring M in the paper
#         self.CNN = nn.Conv1d(in_channels=self.embedding_dim,
#                              out_channels=self.hidden_dim, kernel_size=3, padding=1)
#         self.RELU = nn.ReLU()
#         self.Tanh = nn.Tanh()
#         self.DropOut = nn.Dropout(p=manager.dropout_p)


#     def forward(self, news_batch, **kwargs):
#         """ encode news through 1-d CNN and combine embeddings with personalized attention

#         Args:
#             news_batch: tensor of [batch_size, *, signal_length]

#         Returns:
#             news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, hidden_dim]
#             news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
#         """
#         e_u = self.DropOut(self.user_embedding(kwargs['user_index']))
#         word_query = self.Tanh(self.wordQueryProject(
#             self.RELU(self.wordPrefProject(e_u))))

#         news_embedding_pretrained = self.DropOut(self.embedding(
#             news_batch)).view(-1, news_batch.shape[-1], self.embedding_dim).transpose(-2, -1)
#         news_embedding = self.RELU(self.CNN(
#             news_embedding_pretrained)).transpose(-2, -1).view(news_batch.shape + (self.hidden_dim,))

#         news_repr = Attention.scaled_dp_attention(word_query.view(
#             word_query.shape[0], 1, 1, word_query.shape[-1]), news_embedding, news_embedding).squeeze(dim=-2)
#         return news_embedding.view(news_batch.shape + (self.level, self.hidden_dim)), news_repr