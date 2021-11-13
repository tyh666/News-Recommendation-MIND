import math
import torch
import torch.nn as nn
from transformers.modeling_utils import apply_chunking_to_forward
from transformers.activations import ACT2FN
from transformers import BertConfig
from .Attention import XSoftmax, scaled_dp_attention

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.all_head_size = config.all_head_size

        self.query = nn.Linear(config.hidden_size, config.all_head_size)
        self.key = nn.Linear(config.hidden_size, config.all_head_size)
        self.value = nn.Linear(config.hidden_size, config.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(
        self,
        hidden_states,
        attention_mask=None
    ):
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        # B, L, L
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        outputs = (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.all_head_size, config.all_head_size)
        self.LayerNorm = nn.LayerNorm(config.all_head_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,)
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.all_head_size, 4 * config.all_head_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(4 * config.all_head_size, config.all_head_size)
        self.LayerNorm = nn.LayerNorm(config.all_head_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask
        )
        attention_output = self_attention_outputs[0]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (attention_output,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
