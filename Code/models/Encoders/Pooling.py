import torch
import torch.nn as nn
from ..Modules.Attention import ScaledDpAttention

class Attention_Pooling(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.name = 'attention-pooling'
        self.query = nn.Parameter(torch.randn(1, config.hidden_dim))

    def forward(self, news_reprs):
        """
        encode user history into a representation vector

        Args:
            news_reprs: batch of news representations, [batch_size, *, hidden_dim]

        Returns:
            user_repr: user representation (coarse), [batch_size, 1, hidden_dim]
        """
        user_repr = ScaledDpAttention(self.query, news_reprs, news_reprs).transpose(-1,-2)
        return user_repr

class Average_Pooling(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.name = 'average-pooling'

    def forward(self, news_reprs, *args):
        """
        encode user history into a representation vector

        Args:
            news_reprs: batch of news representations, [batch_size, *, hidden_dim]

        Returns:
            user_repr: user representation (coarse), [batch_size, 1, hidden_dim]
        """
        user_repr = news_reprs.mean(dim=1, keepdim=True)
        return user_repr