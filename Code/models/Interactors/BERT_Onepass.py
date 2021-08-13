import torch
from torch._C import device
import torch.nn as nn
from transformers import BertModel,BertConfig
from .Modules.OnePassAttn import BertSelfAttention

class BERT_Interactor(nn.Module):
    def __init__(self, config):
        # confirm the hidden dim to be 768
        assert config.hidden_dim == 768
        # confirm term_num + signal_length is less than 512
        # assert config.k * config.his_size + config.his_size + config.signal_length < 512

        super().__init__()

        self.name = 'bert-overlook'
        self.signal_length = config.signal_length
        self.term_num = config.term_num
        # self.device = config.device

        self.hidden_dim = config.hidden_dim

        self.final_dim = self.hidden_dim

        bert_config = BertConfig()
        # primary bert
        prim_bert = BertModel(bert_config).encoder
        bert_config.signal_length = config.signal_length
        bert_config.term_num = config.term_num
        bert_config.cdd_size = config.cdd_size
        for l in prim_bert.layer:
            l.attention.self = BertSelfAttention(bert_config)

        bert = BertModel.from_pretrained(
            config.bert,
            cache_dir=config.path + 'bert_cache/'
        )
        prim_bert.load_state_dict(bert.encoder.state_dict())
        self.bert = prim_bert

        # self.inte_embedding = nn.Parameter(torch.randn(1,1,1,self.hidden_dim))
        self.order_embedding = nn.Parameter(torch.randn(1, config.his_size, 1, config.hidden_dim))

        # [1, *, hidden_dim]
        self.cdd_pos_embedding = nn.Parameter(bert.embeddings.position_embeddings.weight[:config.signal_length + 1].unsqueeze(0).unsqueeze(0))
        self.pst_pos_embedding = nn.Parameter(bert.embeddings.position_embeddings.weight[config.signal_length + 1: config.signal_length + 1 + config.term_num].unsqueeze(0))

        # [SEP] token
        self.sep_embedding = nn.Parameter(bert.embeddings.word_embeddings(torch.tensor([102])).clone().detach().requires_grad_(True).view(1,1,self.hidden_dim))
        self.cls_embedding = nn.Parameter(bert.embeddings.word_embeddings(torch.tensor([101])).clone().detach().requires_grad_(True).view(1,1,self.hidden_dim))

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
        # ps_terms = torch.cat([ps_terms, self.inte_embedding.expand(batch_size, self.his_size, 1, self.hidden_dim)], dim=-2)

        # add order embedding and sep embedding
        ps_terms = (ps_terms + self.order_embedding).view(batch_size, -1, self.hidden_dim)
        ps_terms = torch.cat([self.sep_embedding.expand(batch_size, 1, self.hidden_dim), ps_terms], dim=1)
        # add position embedding
        ps_terms += self.pst_pos_embedding
        # insert cls token for pooling
        # ps_terms = torch.cat([self.cls_embedding.expand(batch_size, 1, self.hidden_dim), ps_terms.view(batch_size, -1, self.hidden_dim)], dim=-2)
        return ps_terms


    def forward(self, ps_terms, cdd_news_embedding, cdd_attn_mask):
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

        # [bs,tn,hd]
        ps_terms = self.fusion(ps_terms, batch_size)

        # add [CLS] token
        # [bs, cs, sl+1, hd]
        cdd_news_embedding = torch.cat([self.cls_embedding.expand(batch_size, cdd_size, 1, self.hidden_dim), cdd_news_embedding], dim=-2)
        # [bs, cs*(sl+1), hd]
        cdd_news_embedding = (cdd_news_embedding + self.cdd_pos_embedding).view(batch_size, -1, self.hidden_dim)

        # [CLS], cdd_news, [SEP], his_news_1, his_news_2, ...
        bert_input = torch.cat([cdd_news_embedding, ps_terms], dim=-2)

        # [bs, cs*(sl+1)]
        attn_mask = torch.cat([torch.ones(batch_size, cdd_size, 1, device=cdd_attn_mask.device), cdd_attn_mask], dim=2).view(batch_size, -1)
        attn_mask = torch.cat([attn_mask, torch.ones(batch_size, self.term_num, device=cdd_attn_mask.device)], dim=-1).view(batch_size, 1, 1, -1)

        bert_output = self.bert(bert_input, attention_mask=attn_mask).last_hidden_state[:, 0 : cdd_size * (self.signal_length + 1) : self.signal_length + 1].view(batch_size, cdd_size, self.hidden_dim)

        return bert_output

if __name__ == '__main__':
    from models.Interactors.BERT import BERT_Interactor
    from data.configs.demo import config
    import torch

    config.hidden_dim = 768
    config.k = 3
    config.bert = 'bert-base-uncased'
    config.his_size = 2
    intr = BERT_Interactor(config)

    a = torch.rand(2, 3, 512, 1, config.hidden_dim)
    b = torch.rand(2, 2, 3, 1, config.hidden_dim)

    res = (intr(a,b))
    print(res.shape)