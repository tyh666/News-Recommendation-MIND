import math
import torch
import torch.nn as nn
from .Attention import XSoftmax

class BertSelfAttention(nn.Module):
    def __init__(self, manager):
        """
        one-pass bert, where other candidate news except itself are masked
        """
        super().__init__()

        self.num_attention_heads = manager.num_attention_heads
        self.attention_head_size = int(manager.hidden_size / manager.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(manager.hidden_size, self.all_head_size)
        self.key = nn.Linear(manager.hidden_size, self.all_head_size)
        self.value = nn.Linear(manager.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(manager.attention_probs_dropout_prob)

        self.signal_length = manager.signal_length
        self.all_length = manager.cdd_size * self.signal_length
        self.term_num = manager.term_num
        self.full_attn = manager.full_attn

        # default to term_num = his_size * k + 1
        self.register_buffer('one_pass_attn_mask_train', torch.cat([torch.eye(manager.cdd_size).repeat_interleave(repeats=self.signal_length, dim=-1).repeat_interleave(repeats=self.signal_length, dim=0), torch.ones(manager.cdd_size * self.signal_length, manager.term_num)], dim=-1).unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer('one_pass_attn_mask_eval', torch.eye(manager.impr_size).repeat_interleave(repeats=self.signal_length, dim=-1), persistent=False)
        self.register_buffer('ps_term_mask', torch.ones(1,self.term_num), persistent=False)

    def transpose_for_scores(self, x):
        """
        transpose the head_num dimension, to make every head operates in parallel
        """
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
        """ customized bert self attention, attending to the references

        Args:
            hidden_states: normally encoded candidate news, [batch_size, signal_length, hidden_dim]
            references: normally personalized terms, [batch_size, term_num, hidden_dim]
        """
        if self.training:
            one_pass_mask = self.one_pass_attn_mask_train
        else:
            attn_field_length = hidden_states.size(1) - self.term_num
            cdd_size, extra = divmod(attn_field_length, self.signal_length)
            assert extra == 0
            one_pass_mask = torch.cat([(self.one_pass_attn_mask_eval[:cdd_size, :cdd_size * self.signal_length]).repeat_interleave(repeats=self.signal_length, dim=0), self.ps_term_mask.expand(attn_field_length, self.term_num)], dim=-1).unsqueeze(0).unsqueeze(0)

        attn_field = hidden_states[:, :-self.term_num]

        # [batch_size, head_num, *, head_dim]
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        cdd_layer = self.transpose_for_scores(self.query(attn_field))
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(cdd_layer, key_layer.transpose(-1, -2))
        # [bs, hn, cdd_length, *]
        attention_scores = (attention_scores / math.sqrt(self.attention_head_size))
        attention_mask_query = one_pass_mask * attention_mask[:, :, :-self.term_num]
        # attention_mask_query = (1 - one_pass_mask * attention_mask) * -10000.

        # Normalize the attention scores to probabilities.
        attention_probs = XSoftmax.apply(attention_scores, attention_mask_query, -1)
        # attention_probs = attention_scores + attention_mask_query
        attention_probs = self.dropout(attention_probs)

        # full attention
        if self.full_attn:
            pst_layer = self.transpose_for_scores(self.query(hidden_states[:, -self.term_num:]))
            attention_scores_pst = torch.matmul(pst_layer, pst_layer.transpose(-1, -2))
            attention_scores_pst = attention_scores_pst / math.sqrt(self.attention_head_size)
            attention_mask_pst = attention_mask[:, :, -self.term_num:, -self.term_num:]
            attention_probs_pst = XSoftmax.apply(attention_scores_pst, attention_mask_pst, -1)
            attention_probs_pst = self.dropout(attention_probs_pst)
            context_layer = torch.cat([torch.matmul(attention_probs, value_layer), torch.matmul(attention_probs_pst, value_layer[:, :, -self.term_num:])], dim=-2)

        # partial attention, where ps_terms do not interact with each other
        else:
            context_layer = torch.cat([torch.matmul(attention_probs, value_layer), value_layer[:, :, -self.term_num:]], dim=-2)

        # [batch_size, signal_length, head_num, head_dim]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return (context_layer,)