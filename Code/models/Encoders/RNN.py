import torch
import torch.nn as nn

class RNN_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = 'rnn-n'

        self.embedding_dim = config.embedding_dim
        # dimension for the final output embedding/representation
        self.hidden_dim = config.hidden_dim

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, news_embedding):
        """ encode news by RNN

        Args:
            news_embedding: tensor of [batch_size, *, signal_length, embedding_dim]

        Returns:
            encoded_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """

        batch_size = news_embedding.size(0)
        signal_length = news_embedding.size(2)
        # conpress news_num into batch_size
        encoded_embedding = news_embedding.view(-1, *news_embedding.shape[2:])
        encoded_embedding, output = self.lstm(encoded_embedding)
        encoded_embedding = encoded_embedding.view(batch_size, -1, signal_length, 2, self.hidden_dim).mean(dim=-2)
        news_repr = torch.mean(output[0],dim=0).view(batch_size, -1, self.hidden_dim)

        return encoded_embedding, news_repr


class RNN_User_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = 'rnn-u'

        self.hidden_dim = config.hidden_dim
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)

    def forward(self, news_reprs):
        """
        encode user history into a representation vector

        Args:
            news_reprs: batch of news representations, [batch_size, *, hidden_dim]

        Returns:
            user_repr: user representation (coarse), [batch_size, 1, hidden_dim]
        """
        _, user_repr = self.lstm(news_reprs.flip(dims=[1]))
        return user_repr[0].transpose(0,1)