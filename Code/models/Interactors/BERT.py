import torch
from torch._C import device
import torch.nn as nn
from transformers import BertModel,BertConfig


class BERT_Onepass_Interactor(nn.Module):
    """
    one-pass bert: cdd1 cdd2 ... cddn [SEP] pst1 pst2 ...
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

        # extra [SEP] token
        self.term_num = config.term_num + 1
        # self.device = config.device

        self.embedding_dim = config.embedding_dim
        self.final_dim = self.embedding_dim

        bert_config = BertConfig()
        # primary bert
        prim_bert = BertModel(bert_config).encoder
        bert_config.signal_length = self.signal_length
        bert_config.term_num = self.term_num
        bert_config.cdd_size = config.cdd_size
        for l in prim_bert.layer:
            l.attention.self = BertSelfAttention(bert_config)

        bert = BertModel.from_pretrained(
            config.bert,
            cache_dir=config.path + 'bert_cache/'
        )
        prim_bert.load_state_dict(bert.encoder.state_dict())
        self.bert = prim_bert

        # self.inte_embedding = nn.Parameter(torch.randn(1,1,1,self.embedding_dim))
        self.order_embedding = nn.Parameter(torch.randn(1, config.his_size, 1, config.embedding_dim))

        # [1, *, embedding_dim]
        self.cdd_pos_embedding = nn.Parameter(bert.embeddings.position_embeddings.weight[:self.signal_length].unsqueeze(0).unsqueeze(0))
        self.pst_pos_embedding = nn.Parameter(bert.embeddings.position_embeddings.weight[self.signal_length: self.signal_length + self.term_num].unsqueeze(0))

        # [SEP] token
        self.sep_embedding = nn.Parameter(bert.embeddings.word_embeddings(torch.tensor([102])).clone().detach().requires_grad_(True).view(1,1,self.embedding_dim))

        # nn.init.xavier_normal_(self.inte_embedding)
        nn.init.xavier_normal_(self.order_embedding)



    def fusion(self, ps_terms, batch_size):
        """
        fuse the personalized terms, add interval embedding and order embedding

        Args:
            ps_terms: [batch_size, his_size, k, embedding_dim]

        Returns:
            ps_terms: [batch_size, term_num (his_size*k (+ his_size)), embedding_dim]
        """

        # insert interval embedding between historical news
        # [bs,hs,k+1,hd]
        # ps_terms = torch.cat([ps_terms, self.inte_embedding.expand(batch_size, self.his_size, 1, self.embedding_dim)], dim=-2)

        # add order embedding and sep embedding
        ps_terms = (ps_terms + self.order_embedding).view(batch_size, -1, self.embedding_dim)
        ps_terms = torch.cat([self.sep_embedding.expand(batch_size, 1, self.embedding_dim), ps_terms], dim=1)
        # add position embedding
        ps_terms += self.pst_pos_embedding
        # insert cls token for pooling
        # ps_terms = torch.cat([self.cls_embedding.expand(batch_size, 1, self.embedding_dim), ps_terms.view(batch_size, -1, self.embedding_dim)], dim=-2)
        return ps_terms


    def forward(self, cdd_news_embedding, ps_terms, cdd_attn_mask):
        """
        calculate interaction tensor and reduce it to a vector

        Args:
            cdd_news_embedding: word-level representation of candidate news, [batch_size, cdd_size, signal_length, embedding_dim]
            ps_terms: personalized terms, [batch_size, his_size, k, embedding_dim]
            cdd_attn_mask: attention mask of the candidate news, [batch_size, cdd_size, signal_length]

        Returns:
            reduced_tensor: output tensor after CNN2d, [batch_size, cdd_size, final_dim]
        """
        batch_size = cdd_news_embedding.size(0)
        cdd_size = cdd_news_embedding.size(1)

        # [bs,tn,hd]
        ps_terms = self.fusion(ps_terms, batch_size)

        # [bs, cs*sl, hd]
        cdd_news_embedding = (cdd_news_embedding + self.cdd_pos_embedding).view(batch_size, -1, self.embedding_dim)

        bert_input = torch.cat([cdd_news_embedding, ps_terms], dim=-2)

        # [bs, cs*sl]
        attn_mask = cdd_attn_mask.view(batch_size, -1)
        attn_mask = torch.cat([attn_mask, torch.ones(batch_size, self.term_num, device=cdd_attn_mask.device)], dim=-1).view(batch_size, 1, 1, -1)

        bert_output = self.bert(bert_input, attention_mask=attn_mask).last_hidden_state[:, 0 : cdd_size * (self.signal_length) : self.signal_length].view(batch_size, cdd_size, self.embedding_dim)

        return bert_output


class BERT_Overlook_Interactor(nn.Module):
    """
    overlook bert, batched version of onepass bert:
        cdd1 [SEP] pst1 pst2 ...
        cdd2 [SEP] pst1 pst2 ...
        ...
        cddn [SEP] pst1 pst2 ...
    """
    def __init__(self, config):
        from .Modules.OverlookAttn import BertSelfAttention
        # confirm the hidden dim to be 768
        assert config.embedding_dim == 768
        # confirm term_num + signal_length is less than 512
        assert config.term_num + config.signal_length < 512

        super().__init__()

        self.name = 'overlook-bert'

        # extra [SEP] token
        self.term_num = config.term_num + 1
        self.signal_length = config.signal_length

        self.embedding_dim = config.embedding_dim
        self.final_dim = self.embedding_dim

        bert_config = BertConfig()
        # primary bert
        prim_bert = BertModel(bert_config).encoder
        bert_config.signal_length = config.signal_length
        for l in prim_bert.layer:
            l.attention.self = BertSelfAttention(bert_config)

        bert = BertModel.from_pretrained(
            config.bert,
            cache_dir=config.path + 'bert_cache/'
        )
        prim_bert.load_state_dict(bert.encoder.state_dict())
        self.bert = prim_bert

        # self.inte_embedding = nn.Parameter(torch.randn(1,1,1,self.embedding_dim))
        self.order_embedding = nn.Parameter(torch.randn(1, config.his_size, 1, config.hidden_dim))

        self.cdd_pos_embedding = nn.Parameter(bert.embeddings.position_embeddings.weight[:self.signal_length].unsqueeze(0).unsqueeze(0))
        self.pst_pos_embedding = nn.Parameter(bert.embeddings.position_embeddings.weight[self.signal_length: self.signal_length + self.term_num].unsqueeze(0))
        # [SEP] token
        self.sep_embedding = nn.Parameter(bert.embeddings.word_embeddings(torch.tensor([102])).clone().detach().requires_grad_(True).view(1,1,self.embedding_dim))
        self.cls_embedding = nn.Parameter(bert.embeddings.word_embeddings(torch.tensor([101])).clone().detach().requires_grad_(True).view(1,1,self.embedding_dim))

        # nn.init.xavier_normal_(self.inte_embedding)
        nn.init.xavier_normal_(self.order_embedding)



    def fusion(self, ps_terms, batch_size):
        """
        fuse the personalized terms, add interval embedding and order embedding

        Args:
            ps_terms: [batch_size, his_size, k, hidden_dim]

        Returns:
            ps_terms: [batch_size, term_num (his_size*k (+ his_size)), hidden_dim]
        """

        # insert interval embedding between historical news
        # [bs,hs,k+1,hd]
        # ps_terms = torch.cat([ps_terms, self.inte_embedding.expand(batch_size, self.his_size, 1, self.embedding_dim)], dim=-2)

        # add order embedding
        ps_terms = (ps_terms + self.order_embedding).view(batch_size, -1, self.embedding_dim)
        ps_terms = torch.cat([self.sep_embedding.expand(batch_size, 1, self.embedding_dim), ps_terms], dim=1)
        # add position embedding
        ps_terms += self.pst_pos_embedding
        # insert cls token for pooling
        # ps_terms = torch.cat([self.cls_embedding.expand(batch_size, 1, self.embedding_dim), ps_terms.view(batch_size, -1, self.embedding_dim)], dim=-2)
        return ps_terms


    def forward(self, cdd_news_embedding, ps_terms, cdd_attn_mask):
        """
        calculate interaction tensor and reduce it to a vector

        Args:
            ps_terms: personalized terms, [batch_size, his_size, k, hidden_dim]
            cdd_news_embedding: word-level representation of candidate news, [batch_size, cdd_size, signal_length, hidden_dim]
            cdd_attn_mask: attention mask of the candidate news, [batch_size, cdd_size, signal_length]

        Returns:
            reduced_tensor: output tensor after CNN2d, [batch_size, cdd_size, final_dim]
        """
        batch_size = cdd_news_embedding.size(0)
        cdd_size = cdd_news_embedding.size(1)
        bs = batch_size * cdd_size

        # [bs,tn,hd]
        ps_terms = self.fusion(ps_terms, batch_size).unsqueeze(1).expand(batch_size, cdd_size, self.term_num, self.embedding_dim).reshape(-1, self.term_num, self.embedding_dim)

        # [CLS], cdd_news, [SEP], his_news_1, his_news_2, ...
        bert_input = torch.cat([cdd_news_embedding.view(bs, -1, self.embedding_dim), ps_terms], dim=-2)
        attn_mask = torch.cat([cdd_attn_mask.view(bs, self.signal_length), torch.ones(bs, self.term_num, device=ps_terms.device)], dim=-1).view(bs, 1, 1, -1)
        bert_output = self.bert(bert_input, attention_mask=attn_mask).last_hidden_state[:, 0].view(batch_size, cdd_size, self.embedding_dim)

        return bert_output


class BERT_Selected_Interactor(nn.Module):
    """
    original Bert, full self-attention, cdd1 cdd2 ... cddn [SEP] his1 [SEP] his2 ...
    """
    def __init__(self, config):
        # confirm the hidden dim to be 768
        assert config.embedding_dim == 768
        assert config.signal_length * (config.impr_size + config.k) + config.impr_size <= 512, "maximum length is {}, exceeding 512".format(config.signal_length * (config.impr_size + config.k) + config.impr_size)

        super().__init__()

        self.name = 'bert'
        self.k = config.k
        self.signal_length = config.signal_length
        self.embedding_dim = config.embedding_dim
        self.final_dim = self.embedding_dim

        bert = BertModel.from_pretrained(
            config.bert,
            cache_dir=config.path + 'bert_cache/'
            )

        self.bert = bert.encoder

        # self.inte_embedding = nn.Parameter(torch.randn(1,1,1,self.embedding_dim))
        self.order_embedding = nn.Parameter(torch.randn(1, 1, self.k, 1, self.embedding_dim))

        # [1, *, embedding_dim]
        self.cdd_pos_embedding = nn.Parameter(bert.embeddings.position_embeddings.weight[:self.signal_length].unsqueeze(0).unsqueeze(0))
        self.pst_pos_embedding = nn.Parameter(bert.embeddings.position_embeddings.weight[self.signal_length: 2 * self.signal_length + 1].unsqueeze(0))

        # [SEP] token
        self.sep_embedding = nn.Parameter(bert.embeddings.word_embeddings(torch.tensor([102])).clone().detach().requires_grad_(True).view(1,1,1,1,self.embedding_dim))

        # nn.init.xavier_normal_(self.inte_embedding)
        nn.init.xavier_normal_(self.order_embedding)


    def fusion(self, his_selected_embedding, batch_size, cdd_size):
        """
        fuse the historical news embeddings, add interval embedding and order embedding

        Args:
            his_selected_embedding: [batch_size, cdd_size, k, signal_length, embedding_dim]

        Returns:
            his_selected_embedding: [batch_size, cdd_size, signal_length * k, embedding_dim]
        """

        # insert interval embedding between historical news
        # [bs,hs,k+1,hd]
        # ps_terms = torch.cat([ps_terms, self.inte_embedding.expand(batch_size, k, 1, self.embedding_dim)], dim=-2)

        # add order embedding
        his_selected_embedding += self.order_embedding
        his_selected_embedding = torch.cat([self.sep_embedding.expand(batch_size, his_selected_embedding.size(1), self.k, 1, self.embedding_dim), his_selected_embedding], dim=-2)

        # insert cls token for pooling
        # ps_terms = torch.cat([self.cls_embedding.expand(batch_size, 1, self.embedding_dim), ps_terms.view(batch_size, -1, self.embedding_dim)], dim=-2)
        return his_selected_embedding.view(batch_size, cdd_size, self.k * (self.signal_length + 1), self.embedding_dim)


    def forward(self, cdd_news_embedding, his_selected_embedding, cdd_attn_mask, his_attn_mask):
        """
        calculate interaction tensor and reduce it to a vector

        Args:
            cdd_news_embedding: word-level representation of candidate news, [batch_size, cdd_size, signal_length, level, embedding_dim]
            his_selected_embedding: word-level representation of selected historical news, [batch_size, cdd_size, k, signal_length, level, embedding_dim]
            cdd_attn_mask: attention mask of the candidate news, [batch_size, cdd_size, signal_length]
            his_attn_mask: attention mask of the candidate news, [batch_size, k, signal_length]

        Returns:
            reduced_tensor: output tensor after CNN2d, [batch_size, cdd_size, final_dim]
        """
        batch_size = cdd_news_embedding.size(0)
        cdd_size = cdd_news_embedding.size(1)
        bs = batch_size * cdd_size

        his_selected_embedding = his_selected_embedding.mean(dim=-2)
        cdd_news_embedding = cdd_news_embedding.mean(dim=-2)

        # [bs,cs,k*sl,ed]
        his_selected_embedding = self.fusion(his_selected_embedding, batch_size, cdd_size)
        cdd_news_embedding += self.cdd_pos_embedding
        bert_input = torch.cat([cdd_news_embedding, his_selected_embedding], dim=-2).view(bs, -1, self.embedding_dim)

        # cdd_news, [SEP], his_news_1, his_news_2, ...
        his_attn_mask_sep = torch.ones((batch_size, cdd_size, self.k, self.signal_length + 1), device=his_selected_embedding.device)
        his_attn_mask_sep[:, :, :, :self.signal_length] = his_attn_mask
        attn_mask = torch.cat([
            cdd_attn_mask.view(bs, self.signal_length),
            his_attn_mask_sep.view(bs, -1)
        ], dim=-1)

        bert_output = self.bert(bert_input, attention_mask=attn_mask.unsqueeze(1).unsqueeze(1)).last_hidden_state[:, 0].view(batch_size, cdd_size, self.embedding_dim)

        return bert_output