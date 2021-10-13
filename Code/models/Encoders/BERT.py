import torch
from torch._C import device
import torch.nn as nn
import re
import torch.nn.functional as F
from transformers import AutoModel
from models.UniLM.configuration_tnlrv3 import TuringNLRv3Config
from models.UniLM.modeling import TuringNLRv3ForSequenceClassification, relative_position_bucket
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

        if manager.bert == 'unilm':
            config = TuringNLRv3Config.from_pretrained(manager.unilm_config_path)
            bert = TuringNLRv3ForSequenceClassification.from_pretrained(manager.unilm_path, config=config).bert

            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            # unique in UniLM
            self.rel_pos_bias = bert.rel_pos_bias

        else:
            bert = AutoModel.from_pretrained(
                manager.get_bert_for_load(),
                cache_dir=manager.path + 'bert_cache/'
            )

        self.bert = bert.encoder


        if manager.bert == 'deberta':
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

        # concatenated ps_terms
        if signal_length > self.signal_length:
            # add [CLS] and [SEP] to ps_terms
            bert_input = torch.cat([self.bert_cls_embedding.expand(bs, 1, self.hidden_dim), bert_input, self.bert_sep_embedding.expand(bs, 1, self.hidden_dim)], dim=-2)
            attn_mask = torch.cat([self.extra_attn_mask.expand(bs, 1), attn_mask, self.extra_attn_mask.expand(bs, 1)], dim=-1)
            signal_length += 2

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

        if hasattr(self, 'rel_pos_bias'):
            position_ids = torch.arange(signal_length, dtype=torch.long, device=bert_input.device).unsqueeze(0).expand(bs, signal_length)
            rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
            rel_pos = relative_position_bucket(rel_pos_mat, num_buckets=self.rel_pos_bins, max_distance=self.max_rel_pos)
            rel_pos = F.one_hot(rel_pos, num_classes=self.rel_pos_bins).float()
            rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
            bert_output = self.bert(bert_input, attention_mask=ext_attn_mask, rel_pos=rel_pos)[0]

        else:
            # [bs, sl/term_num+2, hd]
            bert_output = self.bert(bert_input, attention_mask=ext_attn_mask).last_hidden_state

        # news_repr = bert_output[:, 0].reshape(batch_size, -1, self.hidden_dim)
        news_repr = scaled_dp_attention(self.query, bert_output, bert_output, attn_mask=attn_mask.unsqueeze(1)).view(batch_size, -1, self.hidden_dim)

        news_encoded_embedding = bert_output.view(batch_size, -1, bert_input.size(-2), self.hidden_dim)

        return news_encoded_embedding, news_repr