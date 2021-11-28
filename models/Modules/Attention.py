import math
import torch
from torch import _softmax_backward_data, nn

def scaled_dp_attention(query, key, value, attn_mask=None):
    """ calculate scaled attended output of values

    Args:
        query: tensor of [batch_size, *, query_num, key_dim]
        key: tensor of [batch_size, *, key_num, key_dim]
        value: tensor of [batch_size, *, key_num, value_dim]
        attn_mask: tensor of [batch_size, *, query_num, key_num]

    Returns:
        attn_output: tensor of [batch_size, *, query_num, value_dim]
    """

    # make sure dimension matches
    assert query.shape[-1] == key.shape[-1]
    key = key.transpose(-2, -1)

    attn_score = torch.matmul(query, key)/math.sqrt(query.shape[-1])

    if attn_mask is not None:
        attn_prob = XSoftmax.apply(attn_score, attn_mask, -1)
    else:
        attn_prob = torch.softmax(attn_score, -1)

    attn_output = torch.matmul(attn_prob, value)
    return attn_output


def get_attn_mask(attn_mask):
    """
    extend the attention mask

    Args:
        attn_mask: [batch_size, *]

    Returns:
        attn_mask: [batch_size, 1, *, *]
    """
    if attn_mask is None:
        return None

    assert attn_mask.dim() == 2

    extended_attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
    extended_attn_mask2 = extended_attn_mask.squeeze(-2).unsqueeze(-1)

    attn_mask = extended_attn_mask * extended_attn_mask2

    return attn_mask


class XSoftmax(torch.autograd.Function):
    """
    Masked Softmax which is optimized for saving memory

    Args:
        input (:obj:`torch.tensor`): The input tensor that will apply softmax.
        mask (:obj:`torch.IntTensor`): The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
    """

    @staticmethod
    def forward(self, input, mask, dim):
        self.dim = dim
        rmask = ~(mask.bool())

        output = input.masked_fill(rmask, float("-inf"))
        output = torch.softmax(output, self.dim)
        output.masked_fill_(rmask, 0)
        self.save_for_backward(output)
        return output

    @staticmethod
    def backward(self, grad_output):
        (output,) = self.saved_tensors
        inputGrad = _softmax_backward_data(grad_output, output, self.dim, output)
        return inputGrad, None, None


class MultiheadAttention(nn.Module):
    def __init__(self, hidden_dim, head_num, key_dim=None, value_dim=None):
        super().__init__()
        self.head_num = head_num
        if not (key_dim and value_dim):
            assert hidden_dim % head_num == 0, "hidden_dim {} must divide head_num {}".format(hidden_dim, head_num)
            head_dim = hidden_dim // head_num
        self.hidden_dim = hidden_dim

        if key_dim:
            self.key_dim = key_dim
        else:
            self.key_dim = head_dim
        if value_dim:
            self.value_dim = value_dim
        else:
            self.value_dim = head_dim

        self.keyProject = nn.Linear(hidden_dim, self.key_dim * head_num)
        self.valueProject = nn.Linear(hidden_dim, self.value_dim * head_num)

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
            hidden_states: normally encoded candidate news, [batch_size, signal_length, hidden_dim]

        Returns:
            attn_output: [batch_size, signal_length, value_dim * num_head]
        """
        # [batch_size, head_num, *, head_dim]
        query = self.transpose_for_scores(self.keyProject(hidden_states))
        key = self.transpose_for_scores(self.keyProject(hidden_states))
        value = self.transpose_for_scores(self.valueProject(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query, key.transpose(-1, -2))

        # [bs, hn, sl+1, *]
        attention_scores = (attention_scores / math.sqrt(self.key_dim))

        if attention_mask is not None:
            attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
        else:
            attention_probs = torch.softmax(attention_scores, -1)

        attn_output = torch.matmul(attention_probs, value)

        # [batch_size, signal_length, head_num, head_dim]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        new_shape = attn_output.size()[:-2] + (self.value_dim * self.head_num,)
        attn_output = attn_output.view(*new_shape)

        return attn_output
