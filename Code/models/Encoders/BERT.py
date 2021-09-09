import torch.nn as nn
from transformers import AutoModel

class BERT_Encoder(nn.Module):
    """
        bert encoder
    """
    def __init__(self, config):
        super().__init__()
        bert_map = {
            'random':'bert',
            "bert":"bert",
            "deberta":"deberta"
        }
        self.name = bert_map[config.embedding]

        # dimension for the final output embedding/representation
        self.hidden_dim = 768

        bert = AutoModel.from_pretrained(
            config.bert,
            cache_dir=config.path + 'bert_cache/'
        )
        self.bert = bert.encoder


    def get_attn_mask(self, attn_mask):
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
        attn_mask = extended_attn_mask * extended_attn_mask.squeeze(-2).unsqueeze(-1)
        attn_mask = attn_mask.byte()

        return attn_mask


    def forward(self, news_embedding, attn_mask):
        """ encode news with bert

        Args:
            news_embedding: [batch_size, *, signal_length, embedding_dim]
            attn_mask: [batch_size, *, signal_length]

        Returns:
            news_encoded_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, emedding_dim]
            news_repr: news representation, of size [batch_size, *, embedding_dim]
        """
        batch_size = news_embedding.size(0)
        signal_length = news_embedding.size(2)

        bert_input = news_embedding.view(-1, signal_length, self.hidden_dim)
        attn_mask = self.get_attn_mask(attn_mask)

        if self.name == 'bert':
            attn_mask = (1.0 - attn_mask) * -10000.0

        # [bs, cs/hs, sl, ed]
        bert_output = self.bert(bert_input, attention_mask=attn_mask).last_hidden_state
        news_repr = bert_output[:, 0].reshape(batch_size, -1, self.hidden_dim)
        # news_repr = self.pooler(bert_output[:, 0].reshape(batch_size, -1, self.hidden_dim))

        news_encoded_embedding = bert_output.view(batch_size, -1, signal_length, self.hidden_dim)

        return news_encoded_embedding, news_repr