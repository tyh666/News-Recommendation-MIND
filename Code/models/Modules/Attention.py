import math
import torch
import torch.nn as nn

def scaled_dp_attention(query, key, value):
    """ calculate scaled attended output of values

    Args:
        query: tensor of [batch_size, *, query_num, key_dim]
        key: tensor of [batch_size, *, key_num, key_dim]
        value: tensor of [batch_size, *, key_num, value_dim]

    Returns:
        attn_output: tensor of [batch_size, *, query_num, value_dim]
    """

    # make sure dimension matches
    assert query.shape[-1] == key.shape[-1]
    key = key.transpose(-2, -1)

    attn_weights = torch.matmul(query, key)/math.sqrt(query.shape[-1])
    attn_weights = nn.functional.softmax(attn_weights,dim=-1)

    attn_output = torch.matmul(attn_weights, value)
    return attn_output


def get_attn_mask(attn_mask, query_length=None):
    """
    extend the attention mask

    Args:
        attn_mask: [batch_size, *]

    Returns:
        attn_mask: [batch_size, 1, *, *]
    """
    if attn_mask.dim() == 3:
        attn_mask = attn_mask.view(-1, attn_mask.size(-1))

    extended_attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
    if query_length is not None:
        extended_attn_mask2 = extended_attn_mask.squeeze(-2).unsqueeze(-1)[:, :, :query_length]
    else:
        extended_attn_mask2 = extended_attn_mask.squeeze(-2).unsqueeze(-1)

    attn_mask = extended_attn_mask * extended_attn_mask2
    attn_mask = attn_mask.byte()

    return attn_mask


class MultiheadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num, key_dim=None, value_dim=None):
        super().__init__()
        self.head_num = head_num
        if not (key_dim and value_dim):
            assert embedding_dim % head_num == 0, "embedding_dim {} must divide head_num {}".format(embedding_dim, head_num)
            head_dim = embedding_dim // head_num
        self.embedding_dim = embedding_dim

        if key_dim:
            self.key_dim = key_dim
        else:
            self.key_dim = head_dim
        if value_dim:
            self.value_dim = value_dim
        else:
            self.value_dim = head_dim

        self.queryProject = nn.Linear(embedding_dim, self.key_dim * head_num)
        self.keyProject = nn.Linear(embedding_dim, self.key_dim * head_num)
        self.valueProject = nn.Linear(embedding_dim, self.value_dim * head_num)

        self.softMax = nn.Softmax(dim=-1)

        nn.init.xavier_normal_(self.queryProject.weight)
        nn.init.xavier_normal_(self.keyProject.weight)
        nn.init.xavier_normal_(self.valueProject.weight)

    def transpose_for_scores(self, x):
        """
        transpose the head_num dimension, to make every head operates in parallel
        """
        new_x_shape = x.size()[:-1] + (self.head_num, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        """ customized bert self attention, attending to the references

        Args:
            hidden_states: normally encoded candidate news, [batch_size, signal_length, embedding_dim]

        Returns:
            attn_output: [batch_size, signal_length, value_dim * num_head]
        """
        # [batch_size, head_num, *, head_dim]
        query = self.transpose_for_scores(self.queryProject(hidden_states))
        key = self.transpose_for_scores(self.keyProject(hidden_states))
        value = self.transpose_for_scores(self.valueProject(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query, key.transpose(-1, -2))

        # [bs, hn, sl+1, *]
        attention_scores = (attention_scores / math.sqrt(self.key_dim))

        if attention_mask:
            attention_scores = attention_scores * attention_mask

        attention_probs = self.softMax(attention_scores)

        attn_output = torch.matmul(attention_probs, value)

        # [batch_size, signal_length, head_num, head_dim]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        new_shape = attn_output.size()[:-2] + (self.value_dim * self.head_num,)
        attn_output = attn_output.view(*new_shape)

        return attn_output
