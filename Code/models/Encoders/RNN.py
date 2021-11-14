import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class RNN_Encoder(nn.Module):
    def __init__(self, manager):
        super().__init__()
        self.embedding_dim = manager.embedding_dim
        # dimension for the final output embedding/representation
        self.hidden_dim = manager.hidden_dim

        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)

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
        encoded_embedding, output = self.rnn(encoded_embedding)
        encoded_embedding = encoded_embedding.view(batch_size, -1, signal_length, 2, self.hidden_dim).mean(dim=-2)
        news_repr = torch.mean(output[0],dim=0).view(batch_size, -1, self.hidden_dim)

        return encoded_embedding, news_repr


class RNN_User_Encoder(nn.Module):
    def __init__(self, manager):
        super().__init__()
        self.hidden_dim = manager.hidden_dim
        self.descend_history = manager.descend_history

        if manager.encoderU == 'gru':
            self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        elif manager.encoderU == 'lstm':
            self.rnn = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)

    def forward(self, news_repr, **kwargs):
        """
        encode user history into a representation vector

        Args:
            news_repr: batch of news representations, [batch_size, *, hidden_dim]
            news_mask: [batch_size, *, 1]

        Returns:
            user_repr: user representation (coarse), [batch_size, 1, hidden_dim]
        """
        if self.descend_history:
            news_repr = news_repr.flip(dims=[1])
        # bs
        if "his_mask" in kwargs:
            lens = kwargs['his_mask'].squeeze(-1).sum(dim=-1).cpu()
            rnn_input = pack_padded_sequence(news_repr, lens, batch_first=True, enforce_sorted=False)
        else:
            rnn_input = news_repr

        _, user_repr = self.rnn(rnn_input)
        if type(user_repr) is tuple:
            user_repr = user_repr[0]
        return user_repr.transpose(0,1)


class LSTUR_User_Encoder(nn.Module):
    def __init__(self, manager):
        super().__init__()
        self.hidden_dim = manager.hidden_dim
        self.rnn = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.userEmbedding = nn.Embedding(manager.get_user_num() + 1, self.hidden_dim)
        nn.init.zeros_(self.userEmbedding.weight[0])

        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)

    def forward(self, news_repr, his_mask=None, user_index=None):
        """
        encode user history into a representation vector

        Args:
            news_repr: batch of news representations, [batch_size, *, hidden_dim]

        Returns:
            user_repr: user representation (coarse), [batch_size, 1, hidden_dim]
        """
        batch_size = news_repr.size(0)
        # mask by p
        default_user_index = torch.zeros(batch_size, dtype=torch.long, device=news_repr.device)
        masked_user_index = default_user_index.bernoulli_() * user_index

        _, user_repr = self.rnn(news_repr.flip(dims=[1]), (self.userEmbedding(masked_user_index).unsqueeze(0), torch.zeros(1, batch_size, self.hidden_dim, device=news_repr.device)))
        return user_repr[0].transpose(0,1)