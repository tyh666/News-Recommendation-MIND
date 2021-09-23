import torch
import torch.nn as nn
from ..Modules.Attention import scaled_dp_attention

class CNN_Encoder(nn.Module):
    def __init__(self, manager):
        super().__init__()
        self.name = 'cnn-n'

        self.hidden_dim = manager.hidden_dim
        self.embedding_dim = manager.embedding_dim

        self.wordQueryProject = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.cnn = nn.Conv1d(
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
        nn.init.xavier_normal_(self.cnn.weight)
        nn.init.xavier_normal_(self.query_words)


    def forward(self, news_embedding, attn_mask=None):
        """ encode news through 1-d CNN

        Args:
            news_embedding: tensor of [batch_size, *, signal_length, embedding_dim]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        signal_length = news_embedding.size(2)
        cnn_input = news_embedding.view(-1, signal_length, self.embedding_dim).transpose(-2, -1)
        cnn_output = self.RELU(self.layerNorm(self.cnn(cnn_input).transpose(-2, -1))).view(*news_embedding.shape[:-1], self.hidden_dim)

        news_repr = scaled_dp_attention(self.query_words, self.Tanh(self.wordQueryProject(cnn_output)), cnn_output, attn_mask.unsqueeze(-2)).squeeze(dim=-2)
        return cnn_output, news_repr


class CNN_User_Encoder(nn.Module):
    def __init__(self, manager):
        super().__init__()
        self.name = 'cnn-u'

        self.hidden_dim = manager.hidden_dim
        self.SeqCNN1D = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim//2, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool1d(3,3),
            nn.Conv1d(self.hidden_dim//2, self.hidden_dim//2//2, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool1d(3,3)
        )
        self.userProject = nn.Linear((self.hidden_dim // 2 // 2) * (manager.his_size // 3 // 3), self.hidden_dim)

        nn.init.xavier_normal_(self.SeqCNN1D[0].weight)
        nn.init.xavier_normal_(self.SeqCNN1D[3].weight)
        nn.init.xavier_normal_(self.userProject.weight)

    def forward(self, news_reprs):
        """
        encode user history into a representation vector by 1D CNN and Max Pooling

        Args:
            news_reprs: batch of news representations, [batch_size, *, hidden_dim]

        Returns:
            user_repr: user representation (coarse), [batch_size, 1, hidden_dim]
        """
        batch_size = news_reprs.size(0)
        encoded_reprs = self.SeqCNN1D(news_reprs.transpose(-2,-1)).view(batch_size, -1)
        user_repr = self.userProject(encoded_reprs).unsqueeze(1)
        return user_repr


if __name__ == '__main__':
    from models.Encoders.CNN import CNN_Encoder
    from data.managers.demo import manager

    manager.npratio = 1
    manager.batch_size = 2
    manager.his_size = 2
    manager.k = 3
    manager.embedding = 'bert'
    manager.bert = 'bert-base-uncased'
    manager.signal_length = 512

    manager.embedding_dim = 768
    manager.hidden_dim = 768

    a = torch.rand(2,2,512,768)
    enc = CNN_Encoder(manager)
    res = enc(a)
    print(res[0].shape, res[1].shape)