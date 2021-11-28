import math
import torch
import torch.nn as nn
from .Attention import XSoftmax

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.dense = nn.Sequential(
            nn.Linear(config.hidden_size, config.signal_length),
            nn.ReLU(),
            nn.Linear(config.signal_length, config.signal_length)
        )
        self.value_fn = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # self.attention_scores = nn.Parameter(torch.empty(1, config.signal_length, config.signal_length))
        # nn.init.xavier_uniform_(self.attention_scores)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        hidden_states = self.value_fn(hidden_states)
        # B, L, L
        attention_scores = self.dense(hidden_states)
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        # B, L, D
        context_layer = torch.matmul(attention_probs, hidden_states)

        outputs = (context_layer,)
        return outputs