import torch
import torch.nn as nn

class RNN_Encoder(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.name = 'rnn-encoder'

        self.level = 2

        self.embedding_dim = config.embedding_dim
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors,sparse=config.spadam,freeze=False)

        # dimension for the final output embedding/representation
        self.hidden_dim = 200

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True,bidirectional=True)

    def forward(self, news_batch, **kwargs):
        """ encode news with bert

        Args:
            news_batch: batch of news tokens, of size [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, level, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """

        # conpress news_num into batch_size
        news_embedding_pretrained = self.embedding(news_batch).view(-1, news_batch.shape[-1], self.embedding_dim)
        news_embedding,output = self.lstm(news_embedding_pretrained)
        news_repr = torch.mean(output[0],dim=0).view(news_batch.shape[0],news_batch.shape[1],self.hidden_dim)

        return news_embedding.view(news_batch.shape + (self.level, self.hidden_dim)), news_repr

class RNN_User_Encoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.name = 'rnn-user-encoder'

        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)

    def forward(self, news_reprs):
        """
            encode user history into a representation vector

        Args:
            news_reprs: batch of news representations, [batch_size, *, hidden_dim]

        Returns:
            user_repr: user representation (coarse), [batch_size, hidden_dim]
        """
        _, user_repr = self.lstm(news_reprs)
        return user_repr[0].transpose(0,1)