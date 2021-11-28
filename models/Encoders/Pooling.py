import torch
import torch.nn as nn
from ..Modules.Attention import scaled_dp_attention

class Attention_Pooling(nn.Module):
    def __init__(self, manager):
        super().__init__()

        self.query_news = nn.Parameter(torch.randn(1, manager.hidden_dim))
        nn.init.xavier_normal_(self.query_news)

    def forward(self, news_reprs, his_mask=None, *args, **kargs):
        """
        encode user history into a representation vector

        Args:
            news_reprs: batch of news representations, [batch_size, *, hidden_dim]

        Returns:
            user_repr: user representation (coarse), [batch_size, 1, hidden_dim]
        """
        if his_mask is not None:
            his_mask = his_mask.to(news_reprs.device).transpose(-1,-2)
        user_repr = scaled_dp_attention(self.query_news, news_reprs, news_reprs, attn_mask=his_mask)
        return user_repr


class Average_Pooling(nn.Module):
    def __init__(self, manager):
        super().__init__()

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