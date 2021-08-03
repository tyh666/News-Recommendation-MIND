import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

class TFM(nn.Module):
    def __init__(self, his_size, k):
        super().__init__()
        self.name = "term-fuser"
        self.size = his_size * k

    def forward(self, ps_terms, ps_term_ids, his_news):
        """ De-duplicate personalized terms and keep differentiability. FIXME: more efficient

        Args:
            ps_terms: personalized terms, [batch_size, his_size, k, hidden_dim]
            ps_term_ids: index of personalized terms, [batch_size, his_size, k]
            his_news: word index of historical news, [batch_size, his_size, signal_length]

        Returns:
            fused_terms_all: de-duplicated personalized terms, representations of all same terms are summed, [batch_size, *, ]
        """
        vocab_index = his_news.gather(dim=-1,index=ps_term_ids)
        fused_terms_all = torch.empty(ps_terms.size(0), self.size, ps_terms.size(-1), device=ps_terms.device, requires_grad=True)
        i = 0

        for batch_i,batch_t in zip(vocab_index, ps_term_ids):
            fuser = defaultdict(list)
            for news_i,news_t in zip(batch_i, batch_t):
                for index, term in zip(news_i, news_t):
                    fuser[index.item()].append(term)

            fused_terms = torch.stack([torch.sum(ps_terms[i][v],dim=0) for v in fuser.values()], dim=0).view(-1, ps_terms.size(-1))
            fused_terms_all[i] = F.pad(fused_terms, [0,0,0,self.size - fused_terms.size(0)], "constant", 0)
            i += 1

        return fused_terms_all


if __name__ == '__main__':
    a = torch.rand(2,3,4,5,requires_grad=True)
    a.retain_grad()

    b = torch.empty(2,3,4).random_(0,8).long()
    c = torch.empty(2,3,8).random_(0,10).long()

    tfm = TFM(3,4)
    x = tfm(a,b,c)

    loss = (x**2).sum()
    loss.backward()
    print(a.grad)
