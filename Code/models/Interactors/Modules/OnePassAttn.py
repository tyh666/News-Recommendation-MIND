import math
import torch
from torch._C import device
import torch.nn as nn

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        """
        one-pass bert, where other candidate news except itself are masked
        """
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.signal_length = config.signal_length
        self.all_length = config.cdd_size * self.signal_length
        self.term_num = config.term_num

        # default to term_num = his_size * k + 1
        self.register_buffer('one_pass_attn_mask_train', torch.cat([torch.eye(config.cdd_size).repeat_interleave(repeats=self.signal_length, dim=-1).repeat_interleave(repeats=self.signal_length, dim=0), torch.ones(config.cdd_size * self.signal_length, config.term_num)], dim=-1).unsqueeze(0).unsqueeze(0), persistent=False)
        # self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
        #     self.max_position_embeddings = config.max_position_embeddings
        #     self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)


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
        # [CLS] + signal_length
        if self.training:
            one_pass_mask = self.one_pass_attn_mask_train
        else:
            attn_field_length = hidden_states.size(1) - self.term_num
            cdd_size = attn_field_length // self.signal_length
            one_pass_mask = torch.cat([torch.eye(cdd_size, device=hidden_states.device).repeat_interleave(repeats=self.signal_length, dim=-1).repeat_interleave(repeats=self.signal_length, dim=0), torch.ones(attn_field_length, self.term_num, device=hidden_states.device)], dim=-1).unsqueeze(0).unsqueeze(0)

        attn_field = hidden_states[:, :-self.term_num]

        # [batch_size, head_num, *, head_dim]
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(attn_field))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
        #     seq_length = hidden_states.size()[1]
        #     position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
        #     position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
        #     distance = position_ids_l - position_ids_r
        #     positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
        #     positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

        #     if self.position_embedding_type == "relative_key":
        #         relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
        #         attention_scores = attention_scores + relative_position_scores
        #     elif self.position_embedding_type == "relative_key_query":
        #         relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
        #         relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
        #         attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        # [bs, hn, sl+1, *]
        attention_scores = (attention_scores / math.sqrt(self.attention_head_size)) * one_pass_mask

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            # print(one_pass_mask, attention_mask)
            attention_scores = attention_scores * attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.cat([torch.matmul(attention_probs, value_layer), value_layer[:, :, -self.term_num:]], dim=-2)

        # [batch_size, signal_length, head_num, head_dim]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return (context_layer,)