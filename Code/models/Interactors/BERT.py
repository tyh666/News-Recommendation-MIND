import torch
import torch.nn as nn
from transformers import BertModel

class BERT_Interactor(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.name = 'bert'
        self.his_size = config.his_size
        # must be 728 to use [SEP] token in bert
        assert config.hidden_dim == 768
        self.hidden_dim = config.hidden_dim

        self.final_dim = self.hidden_dim

        bert = BertModel.from_pretrained(config.bert)

        self.bert = bert.encoder


        self.inte_embedding = nn.Parameter(torch.randn(1,1,1,self.hidden_dim))
        self.order_embedding = nn.Parameter(torch.randn(1, config.his_size, 1, config.hidden_dim))
        # [SEP] token
        self.sep_embedding = nn.Parameter(bert.embeddings.word_embeddings(torch.tensor([102])).clone().detach().requires_grad_(True).view(1,1,self.hidden_dim))
        self.cls_embedding = nn.Parameter(bert.embeddings.word_embeddings(torch.tensor([101])).clone().detach().requires_grad_(True).view(1,1,self.hidden_dim))

        nn.init.xavier_normal_(self.inte_embedding)
        nn.init.xavier_normal_(self.order_embedding)



    def fusion(self, ps_terms, batch_size):
        """
        fuse the personalized terms

        Args:
            ps_terms: [batch_size, his_size, k, level, hidden_dim]

        Returns:
            ps_terms: [batch_size, term_num (his_size*k + his_size), hidden_dim]
        """

        # [bs,hs,k+1,hd]
        ps_terms = torch.cat([ps_terms.squeeze(-2), self.inte_embedding.expand(batch_size, self.his_size, 1, self.hidden_dim)], dim=-2)

        # insert order embedding to seperate different historical news
        # [1,hs,1,hd]
        ps_terms += self.order_embedding

        # insert cls token for pooling
        ps_terms = torch.cat([self.cls_embedding.expand(batch_size, 1, self.hidden_dim), ps_terms.view(batch_size, -1, self.hidden_dim)], dim=-2)
        return ps_terms


    def forward(self, cdd_news_embedding, ps_terms):
        """
        calculate interaction tensor and reduce it to a vector

        Args:
            cdd_news_embedding: word-level representation of candidate news, [batch_size, cdd_size, signal_length, level, hidden_dim]
            ps_terms: personalized terms, [batch_size, his_size, k, level, hidden_dim]

        Returns:
            reduced_tensor: output tensor after CNN2d, [batch_size, cdd_size, final_dim]
        """
        batch_size = cdd_news_embedding.size(0)
        cdd_size = cdd_news_embedding.size(1)
        bs = batch_size * cdd_size

        # [bs,tn,hd]
        ps_terms = self.fusion(ps_terms, batch_size)
        term_num = ps_terms.size(1)

        # make sure the concatenated sequence is shorter than 512
        bert_input = torch.cat([ps_terms.unsqueeze(1).expand(batch_size, cdd_size, ps_terms.size(1), self.hidden_dim).reshape(-1, *ps_terms.shape[1:]), self.sep_embedding.expand(bs, 1, self.hidden_dim), cdd_news_embedding.squeeze(-2).view(bs, -1, self.hidden_dim)[:, :512-term_num-1]], dim=-2)
        bert_output = self.bert(bert_input).last_hidden_state[:, 0, :].view(batch_size, cdd_size, self.hidden_dim)

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