import torch.nn as nn
import torch.nn.functional as F

class DRM_Matching(nn.Module):
    """
    basic document reducer: topk
    """
    def __init__(self, k, threshold = -float('inf')):
        super().__init__()

        self.name = "matching-based"

        self.k = k
        self.threshold = threshold

    def forward(self, news_embedding, user_repr):
        """
        Extract words from news text according to the overall user interest

        Args:
            news_embedding: word-level news embedding, [batch_size, his_size, signal_length, level, hidden_dim]
            user_repr: user representation, [batch_size, 1, hidden_dim]

        Returns:
            weighted_pt: weighted embedding for personalized terms, [batch_size, his_size, k, hidden_dim]
            score_kid: index of top k terms in the text, [batch_size, his_size, k]
        """

        news_embedding_mean = news_embedding.mean(dim=-2)
        # [bs, hs, sl]
        scores = F.normalize(news_embedding_mean, dim=-1).matmul(F.normalize(user_repr, dim=-1).transpose(-2,-1).unsqueeze(1)).squeeze(-1)

        score_k, score_kid = scores.topk(dim=-1, k=self.k)
        personalized_terms = news_embedding.gather(dim=-3,index=score_kid.unsqueeze(-1).unsqueeze(-1).expand(score_kid.size() + (news_embedding.size(-2),news_embedding.size(-1))))

        weighted_ps_terms = personalized_terms * (score_k.masked_fill(score_k < self.threshold, 0).unsqueeze(-1).unsqueeze(-1))
        # weighted_ps_terms.retain_grad()
        # print(weighted_ps_terms.grad, weighted_ps_terms.requires_grad)

        return weighted_ps_terms, score_kid

if __name__ == "__main__":
    import torch
    docReducer = DRM_Matching(2)

    a = torch.rand((2,3,4,1,5),requires_grad=True)
    a.retain_grad()
    b = torch.rand(2,1,5,requires_grad=True)
    b.retain_grad()

    c,_ = docReducer(a,b)
    loss = (c**2).sum()
    loss.backward()

    print(a.grad)