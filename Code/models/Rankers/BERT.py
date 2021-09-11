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
    def __init__(self, config):
        assert config.embedding_dim == 768
        super().__init__()

        self.name = 'original-bert'
        self.k = config.k
        self.signal_length = config.signal_length
        self.embedding_dim = config.embedding_dim

        self.final_dim = self.embedding_dim

        bert = BertModel.from_pretrained(
            config.bert,
            cache_dir=config.path + 'bert_cache/'
        )
        self.bert = bert.encoder
        self.pooler = nn.Sequential(
            nn.Linear(self.embedding_dim, self.final_dim),
            nn.Tanh()
        )

        # [2, embedding_ldim]
        self.token_type_embedding = nn.Parameter(bert.embeddings.token_type_embeddings.weight)
        # [SEP] token
        self.sep_embedding = nn.Parameter(bert.embeddings.word_embeddings(torch.tensor([102])).clone().detach().requires_grad_(True).view(1,1,self.embedding_dim))

        self.order_embedding = nn.Parameter(torch.randn(config.his_size, 1, config.embedding_dim))
        nn.init.xavier_normal_(self.order_embedding)
        nn.init.xavier_normal_(self.pooler[0].weight)

    def forward(self, cdd_news_embedding, his_seq, cdd_attn_mask, his_seq_attn_mask):
        """
        calculate interaction tensor and reduce it to a vector

        Args:
            cdd_news_embedding: word-level representation of candidate news, [batch_size, cdd_size, signal_length, embedding_dim]
            his_seq: personalized terms, [batch_size, (cdd_size, k, signal_length)/(term_num), embedding_dim]
            cdd_attn_mask: attention mask of the candidate news, [batch_size, cdd_size, signal_length]
            his_seq_attn_mask: attention mask of the history sequence, [batch_size, (cdd_size, k, signal_length)/(his_size, k)]

        Returns:
            reduced_tensor: output tensor after CNN2d, [batch_size, cdd_size, final_dim]
        """
        batch_size = cdd_news_embedding.size(0)
        cdd_size = cdd_news_embedding.size(1)
        bs = batch_size * cdd_size

        # [bs, seq_length, embedding_dim]
        if his_seq.dim() == 3:
            his_seq = torch.cat([self.sep_embedding.expand(batch_size, 1, self.embedding_dim), his_seq], dim=1)
            his_seq[:,0] += self.token_type_embedding[0]
            his_seq[:,1:] += self.token_type_embedding[1]
            his_seq = his_seq.unsqueeze(1).expand(batch_size, cdd_size, -1, self.embedding_dim).reshape(bs, -1, self.embedding_dim)
            his_seq_attn_mask = his_seq_attn_mask.view(batch_size, 1, -1).expand(batch_size, cdd_size, -1).reshape(bs, -1)

        elif his_seq.dim() == 5:
            his_seq = his_seq.reshape(bs, -1, self.embedding_dim)
            his_seq = torch.cat([self.sep_embedding.expand(bs, 1, self.embedding_dim), his_seq], dim=1)
            his_seq[:,1:] += self.token_type_embedding[1]
            his_seq[:,0] += self.token_type_embedding[0]

            his_seq_attn_mask = his_seq_attn_mask.reshape(bs, -1)

        else:
            raise ValueError("False input his_seq of size {}".format(his_seq.size()))

        his_seq_attn_mask = torch.cat([torch.ones(bs, 1, device=his_seq.device), his_seq_attn_mask], dim=-1)

        # [bs, cs*sl, hd]
        cdd_news_embedding = (cdd_news_embedding + self.token_type_embedding[0]).view(bs, self.signal_length, self.embedding_dim)

        bert_input = torch.cat([cdd_news_embedding, his_seq], dim=-2)

        # [bs, cs*sl]
        attn_mask = torch.cat([cdd_attn_mask.view(bs, -1), his_seq_attn_mask], dim=-1)
        attn_mask = get_attn_mask(attn_mask)
        attn_mask = (1.0 - attn_mask) * -10000.0

        bert_output = self.bert(bert_input, attention_mask=attn_mask).last_hidden_state[:, 0].view(batch_size, cdd_size, self.embedding_dim)
        bert_output = self.pooler(bert_output)

        return bert_output


class BERT_Onepass_Ranker(nn.Module):
    """
    one-pass bert:
        cdd1 cdd2 ... cddn [SEP] pst1 pst2 ...
    """
    def __init__(self, config):
        from .Modules.OnePassAttn import BertSelfAttention
        # confirm the hidden dim to be 768
        assert config.embedding_dim == 768
        # confirm term_num + signal_length is less than 512
        # assert config.k * config.his_size + config.his_size + config.signal_length < 512

        super().__init__()

        self.name = 'onepass-bert'
        self.signal_length = config.signal_length
        self.term_num = config.term_num + 1
        self.embedding_dim = config.embedding_dim
        self.final_dim = self.embedding_dim

        bert_config = BertConfig()
        # primary bert
        prim_bert = BertModel(bert_config).encoder
        bert_config.signal_length = self.signal_length
        bert_config.term_num = config.term_num + 1
        bert_config.cdd_size = config.cdd_size
        bert_config.impr_size = config.impr_size
        bert_config.full_attn = config.full_attn
        for l in prim_bert.layer:
            l.attention.self = BertSelfAttention(bert_config)

        bert = BertModel.from_pretrained(
            config.bert,
            cache_dir=config.path + 'bert_cache/'
        )
        prim_bert.load_state_dict(bert.encoder.state_dict())
        self.bert = prim_bert

        self.pooler = nn.Sequential(
            nn.Linear(self.embedding_dim, self.final_dim),
            nn.Tanh()
        )
        nn.init.xavier_normal_(self.pooler[0].weight)

        # [2, embedding_dim]
        self.token_type_embedding = nn.Parameter(torch.randn(2, self.embedding_dim))
        nn.init.xavier_normal_(self.token_type_embedding)

        # [SEP] token
        if config.embedding == 'bert':
            self.sep_embedding = nn.Parameter(bert.embeddings.word_embeddings(torch.tensor([102])).clone().detach().requires_grad_(True).view(1,1,self.embedding_dim))
        elif config.embedding == 'deberta':
            self.sep_embedding = nn.Parameter(bert.embeddings.word_embeddings(torch.tensor([2])).clone().detach().requires_grad_(True).view(1,1,self.embedding_dim))
        else:
            self.sep_embedding = nn.Parameter(torch.randn(1,1,self.embedding_dim))
            nn.init.xavier_normal_(self.sep_embedding)

        self.register_buffer('sep_attn_mask', torch.ones(1, 1), persistent=False)

    def forward(self, cdd_news_embedding, ps_terms, cdd_attn_mask, ps_term_mask):
        """
        calculate interaction tensor and reduce it to a vector

        Args:
            cdd_news_embedding: word-level representation of candidate news, [batch_size, cdd_size, signal_length, embedding_dim]
            ps_terms: concatenated historical news or personalized terms, [batch_size, term_num, embedding_dim]
            cdd_attn_mask: attention mask of the candidate news, [batch_size, cdd_size, signal_length]
            ps_term_mask: attention mask of the personalized terms, [batch_size, term_num]

        Returns:
            reduced_tensor: output tensor after CNN2d, [batch_size, cdd_size, final_dim]
        """
        batch_size = cdd_news_embedding.size(0)
        cdd_size = cdd_news_embedding.size(1)

        # [bs,tn,hd]
        ps_terms += self.token_type_embedding[1]

        # [bs, cs*sl, hd]
        cdd_news_embedding = cdd_news_embedding.view(batch_size, -1, self.embedding_dim)

        bert_input = torch.cat([cdd_news_embedding, self.sep_embedding.expand(batch_size, 1, self.embedding_dim), ps_terms], dim=-2)
        bert_input[:, :cdd_news_embedding.size(1) + 1] += self.token_type_embedding[0]

        # [bs, cs*sl]
        attn_mask = cdd_attn_mask.view(batch_size, -1)
        cdd_length = attn_mask.size(-1)

        attn_mask = torch.cat([attn_mask, self.sep_attn_mask.expand(batch_size, 1), ps_term_mask], dim=-1)
        attn_mask = get_attn_mask(attn_mask, query_length=cdd_length)
        attn_mask = (1.0 - attn_mask) * -10000.0

        bert_output = self.bert(bert_input, attention_mask=attn_mask).last_hidden_state[:, 0 : cdd_size * (self.signal_length) : self.signal_length].view(batch_size, cdd_size, self.embedding_dim)
        bert_output = self.pooler(bert_output)

        return bert_output