import torch
import torch.nn as nn
import re
from transformers import AutoModel
from models.Modules.Attention import get_attn_mask, scaled_dp_attention

class BERT_Encoder(nn.Module):
    """
        1. insert [CLS] and [SEP] token embedding to input sequences
        2. add position embedding to the sequence, starting from 0 pos
        2. encode news with bert
    """
    def __init__(self, manager):
        super().__init__()

        # dimension for the final output embedding/representation
        self.hidden_dim = 768
        self.signal_length = manager.signal_length

        bert = AutoModel.from_pretrained(
            manager.bert,
            cache_dir=manager.path + 'bert_cache/'
        )
        self.bert = bert.encoder


        if re.search('deberta-', manager.bert):
            self.extend_attn_mask = True
        else:
            self.extend_attn_mask = False

        word_embedding = bert.embeddings.word_embeddings
        self.bert_cls_embedding = nn.Parameter(word_embedding.weight[manager.get_special_token_id('[CLS]')].view(1,1,self.hidden_dim))
        self.bert_sep_embedding = nn.Parameter(word_embedding.weight[manager.get_special_token_id('[SEP]')].view(1,1,self.hidden_dim))

        self.query = nn.Parameter(torch.randn(1, self.hidden_dim))
        nn.init.xavier_normal_(self.query)

        try:
            self.bert_pos_embedding = nn.Parameter(bert.embeddings.position_embeddings.weight)
        except:
            self.bert_pos_embedding = None

        # try:
        #     self.bert_token_type_embedding = nn.Parameter(bert.embeddings.token_type_embeddings.weight)
        # except:
        #     self.bert_token_type_embedding = None

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
        bs = bert_input.size(0)

        attn_mask = attn_mask.view(-1, signal_length)

        # bert_input = torch.cat([self.bert_cls_embedding.expand(bs, 1, self.hidden_dim), bert_input, self.bert_sep_embedding.expand(bs, 1, self.hidden_dim)], dim=-2)
        if signal_length > self.signal_length:
            # add [CLS] and [SEP] to ps_terms
            bert_input = torch.cat([self.bert_cls_embedding.expand(bs, 1, self.hidden_dim), bert_input, self.bert_sep_embedding.expand(bs, 1, self.hidden_dim)], dim=-2)
            attn_mask = torch.cat([self.extra_attn_mask.expand(bs, 1), attn_mask, self.extra_attn_mask.expand(bs, 1)], dim=-1)

        #     if self.bert_token_type_embedding is not None:
        #         bert_input += self.bert_token_type_embedding[1]

        # else:
        #     if self.bert_token_type_embedding is not None:
        #         bert_input += self.bert_token_type_embedding[0]

        if self.bert_pos_embedding is not None:
            bert_input += self.bert_pos_embedding[:bert_input.size(-2)]

        if self.extend_attn_mask:
            ext_attn_mask = attn_mask
        else:
            ext_attn_mask = (1.0 - attn_mask) * -10000.0
            ext_attn_mask = attn_mask.view(bs, 1, 1, -1)

        # [bs, sl/term_num+2, hd]
        bert_output = self.bert(bert_input, attention_mask=ext_attn_mask).last_hidden_state
        # news_repr = bert_output[:, 0].reshape(batch_size, -1, self.hidden_dim)
        news_repr = scaled_dp_attention(self.query, bert_output, bert_output, attn_mask=attn_mask.unsqueeze(1)).view(batch_size, -1, self.hidden_dim)

        news_encoded_embedding = bert_output.view(batch_size, -1, bert_input.size(-2), self.hidden_dim)

        return news_encoded_embedding, news_repr