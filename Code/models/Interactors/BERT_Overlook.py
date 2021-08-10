import torch
import torch.nn as nn
from transformers import BertModel,BertConfig
from .Modules.OverlookAttn import BertSelfAttention


class BERT_Interactor(nn.Module):
    def __init__(self, config):
        # confirm the hidden dim to be 768
        assert config.hidden_dim == 768
        # confirm term_num + signal_length is less than 512
        assert config.k * config.his_size + config.his_size + config.signal_length < 512

        super().__init__()

        self.name = 'bert-overlook'
        self.his_size = config.his_size
        self.device = config.device

        self.hidden_dim = config.hidden_dim

        self.final_dim = self.hidden_dim

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

        # self.inte_embedding = nn.Parameter(torch.randn(1,1,1,self.hidden_dim))
        self.order_embedding = nn.Parameter(torch.randn(1, config.his_size, 1, config.hidden_dim))
        # [SEP] token
        self.sep_embedding = nn.Parameter(bert.embeddings.word_embeddings(torch.tensor([102])).clone().detach().requires_grad_(True).view(1,1,self.hidden_dim))
        self.cls_embedding = nn.Parameter(bert.embeddings.word_embeddings(torch.tensor([101])).clone().detach().requires_grad_(True).view(1,1,self.hidden_dim))

        # nn.init.xavier_normal_(self.inte_embedding)
        nn.init.xavier_normal_(self.order_embedding)



    def fusion(self, ps_terms, batch_size):
        """
        fuse the personalized terms, add interval embedding and order embedding

        Args:
            ps_terms: [batch_size, his_size, k, level, hidden_dim]

        Returns:
            ps_terms: [batch_size, term_num (his_size*k (+ his_size)), hidden_dim]
        """

        ps_terms = ps_terms.squeeze(-2)
        # insert interval embedding between historical news
        # [bs,hs,k+1,hd]
        # ps_terms = torch.cat([ps_terms, self.inte_embedding.expand(batch_size, self.his_size, 1, self.hidden_dim)], dim=-2)

        # add order embedding
        ps_terms = (ps_terms + self.order_embedding).view(batch_size, -1, self.hidden_dim)
        ps_terms = torch.cat([self.sep_embedding.expand(batch_size, 1, self.hidden_dim), ps_terms], dim=1)
        # insert cls token for pooling
        # ps_terms = torch.cat([self.cls_embedding.expand(batch_size, 1, self.hidden_dim), ps_terms.view(batch_size, -1, self.hidden_dim)], dim=-2)
        return ps_terms


    def forward(self, ps_terms, cdd_news_embedding, cdd_attn_mask):
        """
        calculate interaction tensor and reduce it to a vector

        Args:
            ps_terms: personalized terms, [batch_size, his_size, k, level, hidden_dim]
            cdd_news_embedding: word-level representation of candidate news, [batch_size, cdd_size, signal_length, level, hidden_dim]
            cdd_attn_mask: attention mask of the candidate news, [batch_size, cdd_size, signal_length]

        Returns:
            reduced_tensor: output tensor after CNN2d, [batch_size, cdd_size, final_dim]
        """
        batch_size = cdd_news_embedding.size(0)
        cdd_size = cdd_news_embedding.size(1)
        bs = batch_size * cdd_size

        # [bs,tn,hd]
        ps_terms = self.fusion(ps_terms, batch_size)
        term_num = ps_terms.size(1)

        # [CLS], cdd_news, [SEP], his_news_1, his_news_2, ...
        bert_input = torch.cat([self.cls_embedding.expand(bs, 1, self.hidden_dim), cdd_news_embedding.view(bs, -1, self.hidden_dim), ps_terms.unsqueeze(1).expand(batch_size, cdd_size, term_num, self.hidden_dim).reshape(-1, *ps_terms.shape[1:])], dim=-2)
        attn_mask = torch.cat([torch.ones(bs, 1, device=self.device), cdd_attn_mask.view(bs, -1), torch.ones(bs, term_num, device=self.device)], dim=-1).view(bs, 1, 1, -1)
        bert_output = self.bert(bert_input, attention_mask=attn_mask).last_hidden_state[:, 0].view(batch_size, cdd_size, self.hidden_dim)

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