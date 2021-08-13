import torch
import torch.nn as nn

class RNN_Encoder(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.name = 'rnn-encoder'

        self.level = 2

        self.embedding_dim = config.embedding_dim
        # dimension for the final output embedding/representation
        self.hidden_dim = config.hidden_dim

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, news_embedding):
        """ encode news by RNN

        Args:
            news_embedding: tensor of [batch_size, *, signal_length, embedding_dim]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """

        # conpress news_num into batch_size
        news_embeds = news_embedding.view(-1, *news_embedding.shape[2:])
        news_embeds, output = self.lstm(news_embeds)
        news_repr = torch.mean(output[0],dim=0).view(*news_embedding.shape[0:2],self.hidden_dim)

        return news_embeds.view(*news_embedding.shape[:-1], self.level, self.hidden_dim), news_repr

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
        _, user_repr = self.lstm(news_reprs)
        return user_repr[0].transpose(0,1)