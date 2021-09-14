import torch
import torch.nn as nn
import re
from transformers import AutoModel
from models.Modules.Attention import get_attn_mask

class BERT_Encoder(nn.Module):
    """
        1. insert [CLS] and [SEP] token embedding to input sequences
        2. add position embedding to the sequence, starting from 0 pos
        2. encode news with bert
    """
    def __init__(self, config):
        super().__init__()

        # dimension for the final output embedding/representation
        self.hidden_dim = 768

        bert = AutoModel.from_pretrained(
            config.bert,
            cache_dir=config.path + 'bert_cache/'
        )
        self.bert = bert.encoder

        if re.search('bert-', config.bert):
            self.extend_attn_mask = False
        else:
            self.extend_attn_mask = True

        word_embedding = bert.embeddings.word_embeddings
        self.cls_embedding = nn.Parameter(word_embedding.weight[config.get_special_token_id('[CLS]')].view(1,1,self.hidden_dim))
        self.sep_embedding = nn.Parameter(word_embedding.weight[config.get_special_token_id('[SEP]')].view(1,1,self.hidden_dim))

        try:
            self.pos_embedding = nn.Parameter(bert.embeddings.position_embeddings.weight)
        except:
            self.pos_embedding = None

        self.register_buffer('extra_attn_mask', torch.ones(1, 1), persistent=False)

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
        signal_length = news_embedding.size(-2)

        # insert [CLS] and [SEP] token
        bert_input = news_embedding.view(-1, signal_length, self.hidden_dim)
        cdd_length = bert_input.size(0)

        bert_input = torch.cat([self.cls_embedding.expand(cdd_length, 1, self.hidden_dim), bert_input, self.sep_embedding.expand(cdd_length, 1, self.hidden_dim)], dim=-2)
        if hasattr(self, 'pos_embedding'):
            bert_input += self.pos_embedding[:bert_input.size(1)]

        attn_mask = attn_mask.view(-1, signal_length)
        attn_mask = torch.cat([self.extra_attn_mask.expand(cdd_length, 1), attn_mask, self.extra_attn_mask.expand(cdd_length, 1)], dim=-1)

        if self.extend_attn_mask:
            attn_mask = get_attn_mask(attn_mask)
        else:
            attn_mask = (1.0 - attn_mask) * -10000.0
            attn_mask = attn_mask.view(cdd_length, 1, 1, -1)

        # [bs, cs/hs, sl, ed]
        bert_output = self.bert(bert_input, attention_mask=attn_mask).last_hidden_state
        news_repr = bert_output[:, 0].reshape(batch_size, -1, self.hidden_dim)
        # news_repr = self.pooler(bert_output[:, 0].reshape(batch_size, -1, self.hidden_dim))

        news_encoded_embedding = bert_output.view(batch_size, -1, signal_length + 2, self.hidden_dim)

        return news_encoded_embedding, news_repr