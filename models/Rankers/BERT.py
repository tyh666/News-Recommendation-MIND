import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig, BertModel
from ..Modules.Attention import get_attn_mask, scaled_dp_attention

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

        self.signal_length = manager.signal_length
        self.hidden_dim = manager.bert_dim

        if manager.bert == 'unilm':
            config = TuringNLRv3Config.from_pretrained(manager.unilm_config_path)
            # config.pooler = None
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
        self.layerNorm = bert.embeddings.LayerNorm
        self.dropOut = bert.embeddings.dropout

        self.pooler = manager.pooler
        self.projector = nn.Linear(manager.bert_dim, manager.bert_dim)
        self.activation = manager.get_activation_func()

        self.extend_attn_mask = manager.bert != 'deberta'

        if manager.reducer != 'none':
            word_embedding = bert.embeddings.word_embeddings
            self.bert_cls_embedding = nn.Parameter(word_embedding.weight[manager.get_special_token_id('[CLS]')].view(1,1,self.hidden_dim))
            self.bert_sep_embedding = nn.Parameter(word_embedding.weight[manager.get_special_token_id('[SEP]')].view(1,1,self.hidden_dim))

        if manager.pooler == 'attn':
            self.query = nn.Parameter(torch.randn(1, self.hidden_dim))
            nn.init.xavier_normal_(self.query)

        try:
            self.bert_pos_embedding = nn.Embedding.from_pretrained(bert.embeddings.position_embeddings.weight, freeze=False)
        except:
            self.bert_pos_embedding = None

        try:
            self.bert_token_type_embedding = nn.Parameter(bert.embeddings.token_type_embeddings.weight)
        except:
            self.bert_token_type_embedding = None

        self.register_buffer('extra_attn_mask', torch.ones(1, 1), persistent=False)



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
        signal_length = his_seq.size(-2)

        bs = batch_size * cdd_size

        # [batch_size, term_num, hidden_dim]
        his_seq = his_seq.unsqueeze(1).expand(batch_size, cdd_size, signal_length, self.hidden_dim).reshape(bs, signal_length, self.hidden_dim)
        his_seq_attn_mask = his_seq_attn_mask.unsqueeze(1).expand(batch_size, cdd_size, signal_length).reshape(bs, signal_length)
        # [batch_size, term_num+2, hidden_dim]
        his_seq = torch.cat([self.bert_cls_embedding.expand(bs, 1, self.hidden_dim), his_seq, self.bert_sep_embedding.expand(bs, 1, self.hidden_dim)], dim=-2)
        his_seq_attn_mask = torch.cat([self.extra_attn_mask.expand(bs, 1), his_seq_attn_mask, self.extra_attn_mask.expand(bs, 1)], dim=-1)
        signal_length += 2

        if self.bert_token_type_embedding is not None:
            his_seq += self.bert_token_type_embedding[0]
            cdd_news_embedding = cdd_news_embedding[:, :, 1:] + self.bert_token_type_embedding[1]
        # [bs, sl - 1, hd] strip [CLS]
        cdd_news_embedding = cdd_news_embedding.view(bs, self.signal_length - 1, self.hidden_dim)

        pos_ids = torch.arange(signal_length + self.signal_length - 1, device=cdd_news_embedding.device)
        bert_input = torch.cat([his_seq, cdd_news_embedding], dim=-2)
        bert_input = self.dropOut(self.layerNorm(bert_input + self.bert_pos_embedding(pos_ids)))

        attn_mask = torch.cat([his_seq_attn_mask, cdd_attn_mask.view(bs, -1)[:, 1:]], dim=-1)

        if self.extend_attn_mask:
            ext_attn_mask = (1.0 - attn_mask) * -10000.0
            ext_attn_mask = ext_attn_mask.view(bs, 1, 1, -1)
        else:
            ext_attn_mask = attn_mask

        if hasattr(self, 'rel_pos_bias'):
            position_ids = torch.arange(signal_length, dtype=torch.long, device=bert_input.device).unsqueeze(0).expand(bs, signal_length)
            rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
            rel_pos = relative_position_bucket(rel_pos_mat, num_buckets=self.rel_pos_bins, max_distance=self.max_rel_pos)
            rel_pos = F.one_hot(rel_pos, num_classes=self.rel_pos_bins).float()
            rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
            bert_output = self.bert(bert_input, attention_mask=ext_attn_mask, rel_pos=rel_pos).last_hidden_state

        else:
            # [bs, sl/term_num+2, hd]
            bert_output = self.bert(bert_input, attention_mask=ext_attn_mask).last_hidden_state

        if self.pooler == "cls":
            fusion_tensor = bert_output[:, 0].reshape(batch_size, -1, self.hidden_dim)
        elif self.pooler == "attn":
            fusion_tensor = scaled_dp_attention(self.query, bert_output, bert_output, attn_mask=attn_mask.unsqueeze(1)).view(batch_size, -1, self.hidden_dim)
        elif self.pooler == "avg":
            fusion_tensor = bert_output.mean(dim=-2).reshape(batch_size, -1, self.hidden_dim)

        fusion_tensor = self.activation(self.projector(fusion_tensor))
        return fusion_tensor


class BERT_Onepass_Ranker(nn.Module):
    """
    one-pass bert:
        cdd1 cdd2 ... cddn [SEP] pst1 pst2 ...
    """
    def __init__(self, manager):
        from ..Modules.OnePassAttn import BertSelfAttention

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
        # attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)

        bert_output = self.bert(bert_input, attention_mask=attn_mask).last_hidden_state

        # [CLS] pooler
        output = bert_output[:, 0 : cdd_size * self.signal_length : self.signal_length].view(batch_size, cdd_size, self.hidden_dim)
        # attentive pooler
        # output = scaled_dp_attention(bert_output.view(batch_size, cdd_size, -1, self.hidden_dim),)
        bert_output = self.bert_pooler(output)

        return bert_output


class BERT_Onelayer_Ranker(nn.Module):
    """
    one-layer bert over outputs:
        cdd1 [SEP] his1 [SEP] his2 ...
    """
    def __init__(self, manager):
        from ..Modules.OneLayerBert import BertLayer
        super().__init__()

        self.hidden_dim = manager.bert_dim
        self.signal_length = manager.signal_length

        self.bert = BertLayer(BertConfig())
        self.pooler = manager.pooler
        # project news representations into the same semantic space
        self.projector = nn.Linear(manager.bert_dim, manager.bert_dim)
        self.activation = manager.get_activation_func()

        if manager.pooler == 'attn':
            self.query = nn.Parameter(torch.randn(1, self.hidden_dim))
            nn.init.xavier_normal_(self.query)

        self.extend_attn_mask = manager.bert != 'deberta'


    def forward(self, cdd_news_embedding, ps_terms, cdd_attn_mask, ps_term_mask):
        """
        calculate interaction tensor and reduce it to a vector

        Args:
            cdd_news_embedding: word-level representation of candidate news, [batch_size, cdd_size, signal_length, hidden_dim]
            ps_terms: personalized terms, [batch_size, (cdd_size, k, signal_length)/(term_num), hidden_dim]
            cdd_attn_mask: attention mask of the candidate news, [batch_size, cdd_size, signal_length]
            ps_term_mask: attention mask of the history sequence, [batch_size, (cdd_size, k, signal_length)/term_num]

        Returns:
            reduced_tensor: output tensor after CNN2d, [batch_size, cdd_size, final_dim]
        """
        batch_size = cdd_news_embedding.size(0)
        cdd_size = cdd_news_embedding.size(1)

        cdd_news_embedding = cdd_news_embedding.view(-1, self.signal_length, self.hidden_dim)
        ps_terms = ps_terms.repeat_interleave(repeats=cdd_size, dim=0)
        ps_term_mask = ps_term_mask.repeat_interleave(repeats=cdd_size, dim=0)

        bert_input = torch.cat([ps_terms, cdd_news_embedding], dim=1)
        attn_mask = torch.cat([ps_term_mask, cdd_attn_mask], dim=-1)

        bert_output = self.bert(bert_input, attn_mask)[0]
        if self.pooler == "cls":
            reduced_vec = bert_output[:, 0].reshape(batch_size, -1, self.hidden_dim)
        elif self.pooler == "attn":
            reduced_vec = scaled_dp_attention(self.query, bert_output, bert_output, attn_mask=attn_mask.squeeze(1)).view(batch_size, -1, self.hidden_dim)
        elif self.pooler == "avg":
            reduced_vec = bert_output.mean(dim=-2).reshape(batch_size, -1, self.hidden_dim)
        reduced_vec = self.activation(self.projector(reduced_vec))

        return reduced_vec