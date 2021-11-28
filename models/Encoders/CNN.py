import torch
import torch.nn as nn
from ..Modules.Attention import scaled_dp_attention

class CNN_Encoder(nn.Module):
    def __init__(self, manager):
        super().__init__()

        self.hidden_dim = manager.hidden_dim
        self.embedding_dim = manager.bert_dim

        self.cnn = nn.Conv1d(
            in_channels=self.embedding_dim,
            out_channels=self.hidden_dim,
            kernel_size=3,
            padding=1
        )
        nn.init.xavier_normal_(self.cnn.weight)

        if manager.reducer != "global":
            self.query_words = nn.Parameter(torch.randn(
                (1, self.hidden_dim), requires_grad=True))
            nn.init.xavier_normal_(self.query_words)
            self.wordQueryProject = nn.Linear(self.hidden_dim, self.hidden_dim)
            nn.init.xavier_normal_(self.wordQueryProject.weight)
            self.Tanh = nn.Tanh()

        self.Relu = nn.ReLU()


    def forward(self, news_embedding, attn_mask=None):
        """ encode news through 1-d CNN

        Args:
            news_embedding: tensor of [batch_size, *, signal_length, embedding_dim]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        signal_length, embedding_dim = news_embedding.shape[-2:]
        cnn_input = news_embedding.view(-1, signal_length, self.embedding_dim).transpose(-2, -1)
        cnn_output = self.Relu(self.cnn(cnn_input)).transpose(-2, -1).view(*news_embedding.shape[:-1], self.hidden_dim)

        if hasattr(self, "query_words"):
            if attn_mask is not None:
                news_repr = scaled_dp_attention(self.query_words, self.Tanh(self.wordQueryProject(cnn_output)), cnn_output, attn_mask.unsqueeze(-2)).squeeze(dim=-2)
            else:
                news_repr = scaled_dp_attention(self.query_words, self.Tanh(self.wordQueryProject(cnn_output)), cnn_output).squeeze(dim=-2)
        else:
            news_repr = None
        return cnn_output, news_repr