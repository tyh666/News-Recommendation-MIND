import torch
import torch.nn as nn
from transformers import BertModel,BertConfig
from ..Modules.Attention import get_attn_mask

class BERT_Original_Ranker(nn.Module):
    """
    original bert:
        cdd1 [SEP] his1 [SEP] his2 ...
        cdd2 [SEP] his1 [SEP] his2 ...
        ...
        cddn [SEP] his1 [SEP] his2 ...
    """
    def __init__(self, manager):
        super().__init__()

        self.k = manager.k
        self.signal_length = manager.signal_length
        self.hidden_dim = 768

        bert = BertModel.from_pretrained(
            manager.bert,
            cache_dir=manager.path + 'bert_cache/'
        )
        self.bert = bert.encoder
        self.bert_pooler = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh()
        )
        nn.init.xavier_normal_(self.bert_pooler[0].weight)

        # [2, embedding_ldim]
        self.bert_token_type_embedding = nn.Parameter(bert.embeddings.token_type_embeddings.weight)
        # [SEP] token
        word_embedding = bert.embeddings.word_embeddings
        self.bert_sep_embedding = nn.Parameter(word_embedding.weight[manager.get_special_token_id('[SEP]')].view(1,1,self.hidden_dim))
        self.bert_pos_embedding = nn.Parameter(bert.embeddings.position_embeddings.weight)

        self.register_buffer('extra_sep_mask', torch.zeros(1, 1), persistent=False)


    def forward(self, cdd_news_embedding, his_seq, cdd_attn_mask, his_seq_attn_mask):
        """
        calculate interaction tensor and reduce it to a vector

        Args:
            cdd_news_embedding: word-level representation of candidate news, [batch_size, cdd_size, signal_length, hidden_dim]
            his_seq: personalized terms, [batch_size, (cdd_size, k, signal_length)/(term_num), hidden_dim]
            cdd_attn_mask: attention mask of the candidate news, [batch_size, cdd_size, signal_length]
            his_seq_attn_mask: attention mask of the history sequence, [batch_size, (cdd_size, k, signal_length)/term_num]

        Returns:
            reduced_tensor: output tensor after CNN2d, [batch_size, cdd_size, final_dim]
        """
        batch_size = cdd_news_embedding.size(0)
        cdd_size = cdd_news_embedding.size(1)
        bs = batch_size * cdd_size

        # [batch_size, term_num, hidden_dim]
        if his_seq.dim() == 3:
            his_seq = torch.cat([his_seq, self.bert_sep_embedding.expand(batch_size, 1, self.hidden_dim)], dim=-2)
            his_seq += self.bert_token_type_embedding[1]
            his_seq += self.bert_pos_embedding[self.signal_length: self.signal_length + his_seq.size(1)]

            his_seq = his_seq.unsqueeze(1).expand(batch_size, cdd_size, -1, self.hidden_dim).reshape(bs, -1, self.hidden_dim)
            his_seq_attn_mask = torch.cat([his_seq_attn_mask, self.extra_sep_mask.expand(batch_size, 1)], dim=-1)
            his_seq_attn_mask = his_seq_attn_mask.unsqueeze(-2).expand(batch_size, cdd_size, -1).reshape(bs, -1)

        elif his_seq.dim() == 5:
            his_seq = his_seq.reshape(bs, -1, self.hidden_dim)
            his_seq += self.bert_token_type_embedding[1]

            his_seq_attn_mask = his_seq_attn_mask.reshape(bs, -1)

        else:
            raise ValueError("False input his_seq of size {}".format(his_seq.size()))

        # [bs, cs*sl, hd]
        cdd_news_embedding = (cdd_news_embedding + self.bert_token_type_embedding[0] + self.bert_pos_embedding[:self.signal_length]).view(bs, self.signal_length, self.hidden_dim)

        bert_input = torch.cat([cdd_news_embedding, his_seq], dim=-2)

        # [bs, cs*sl]
        attn_mask = torch.cat([cdd_attn_mask.view(bs, -1), his_seq_attn_mask], dim=-1)
        attn_mask = ((1.0 - attn_mask) * -10000.0).view(bs, 1, 1, -1)

        bert_output = self.bert(bert_input, attention_mask=attn_mask).last_hidden_state[:, 0].view(batch_size, cdd_size, self.hidden_dim)
        bert_output = self.bert_pooler(bert_output)

        return bert_output


class BERT_Onepass_Ranker(nn.Module):
    """
    one-pass bert:
        cdd1 cdd2 ... cddn [SEP] pst1 pst2 ...
    """
    def __init__(self, manager):
        from .Modules.OnePassAttn import BertSelfAttention

        super().__init__()

        self.signal_length = manager.signal_length
        # extra [SEP]
        self.term_num = manager.term_num + 1
        self.hidden_dim = 768

        bert_manager = BertConfig()
        # primary bert
        prim_bert = BertModel(bert_manager).encoder
        # [CLS]
        bert_manager.signal_length = self.signal_length
        # extra [SEP]
        bert_manager.term_num = self.term_num
        bert_manager.cdd_size = manager.cdd_size
        bert_manager.impr_size = manager.impr_size
        bert_manager.full_attn = manager.full_attn
        for l in prim_bert.layer:
            l.attention.self = BertSelfAttention(bert_manager)

        bert = BertModel.from_pretrained(
            manager.bert,
            cache_dir=manager.path + 'bert_cache/'
        )
        prim_bert.load_state_dict(bert.encoder.state_dict())
        self.bert = prim_bert

        self.bert_pooler = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh()
        )
        nn.init.xavier_normal_(self.bert_pooler[0].weight)

        # [2, hidden_dim]
        self.bert_token_type_embedding = nn.Parameter(bert.embeddings.token_type_embeddings.weight)
        # [SEP] token
        word_embedding = bert.embeddings.word_embeddings
        self.bert_sep_embedding = nn.Parameter(word_embedding.weight[manager.get_special_token_id('[SEP]')].view(1,1,self.hidden_dim))
        self.bert_pos_embedding = nn.Parameter(bert.embeddings.position_embeddings.weight)

        self.register_buffer('extra_sep_mask', torch.zeros(1, 1), persistent=False)


    def forward(self, cdd_news_embedding, ps_terms, cdd_attn_mask, ps_term_mask):
        """
        calculate interaction tensor and reduce it to a vector

        Args:
            cdd_news_embedding: word-level representation of candidate news, [batch_size, cdd_size, signal_length, hidden_dim]
            ps_terms: concatenated historical news or personalized terms, [batch_size, term_num, hidden_dim]
            cdd_attn_mask: attention mask of the candidate news, [batch_size, cdd_size, signal_length]
            ps_term_mask: attention mask of the personalized terms, [batch_size, term_num]

        Returns:
            reduced_tensor: output tensor after CNN2d, [batch_size, cdd_size, final_dim]
        """
        batch_size = cdd_news_embedding.size(0)
        cdd_size = cdd_news_embedding.size(1)

        if self.bert_pos_embedding is not None:
            # extra [CLS]
            cdd_news_embedding += self.bert_pos_embedding[:self.signal_length]

        cdd_news_embedding = cdd_news_embedding.view(batch_size, -1, self.hidden_dim)
        cdd_attn_mask = cdd_attn_mask.view(batch_size, -1)

        bert_input = torch.cat([cdd_news_embedding, ps_terms, self.bert_sep_embedding.expand(batch_size, 1, self.hidden_dim)], dim=-2)

        cdd_length = self.signal_length * cdd_size
        bert_input[:, :cdd_length] += self.bert_token_type_embedding[0]
        bert_input[:, cdd_length:] += self.bert_token_type_embedding[1]

        bert_input[:, cdd_length:] += self.bert_pos_embedding[self.signal_length: self.signal_length + self.term_num]

        attn_mask = torch.cat([cdd_attn_mask, ps_term_mask, self.extra_sep_mask.expand(batch_size, 1)], dim=-1)
        attn_mask = get_attn_mask(attn_mask)

        bert_output = self.bert(bert_input, attention_mask=attn_mask).last_hidden_state[:, 0 : cdd_size * self.signal_length : self.signal_length].view(batch_size, cdd_size, self.hidden_dim)
        bert_output = self.bert_pooler(bert_output)

        return bert_output