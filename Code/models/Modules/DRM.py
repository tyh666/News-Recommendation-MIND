import torch
import torch.nn as nn
import torch.nn.functional as F

class Matching_Reducer(nn.Module):
    """
    basic document reducer: topk of each historical article
    """
    def __init__(self, config):
        super().__init__()

        self.name = "matching-reducer"

        self.k = config.k
        self.threshold = config.threshold

        config.term_num = config.k * config.his_size

        if config.threshold != -float('inf'):
            threshold = torch.tensor([config.threshold])
            self.register_buffer('threshold', threshold)

    def forward(self, news_selection_embedding, news_embedding, user_repr, his_attn_mask):
        """
        Extract words from news text according to the overall user interest

        Args:
            news_selection_embedding: encoded word-level embedding, [batch_size, his_size, signal_length, hidden_dim]
            news_embedding: word-level news embedding, [batch_size, his_size, signal_length, hidden_dim]
            user_repr: user representation, [batch_size, 1, hidden_dim]

        Returns:
            weighted_pt: weighted embedding for personalized terms, [batch_size, his_size, k, hidden_dim]
            score_kid: index of top k terms in the text, [batch_size, his_size, k]
        """
        # strip off [CLS]
        news_selection_embedding = news_selection_embedding[:, :, 1:]

        # [bs, hs, sl]
        scores = F.normalize(news_selection_embedding, dim=-1).matmul(F.normalize(user_repr, dim=-1).transpose(-2,-1).unsqueeze(1)).squeeze(-1)
        # mask the padded term
        scores = scores * his_attn_mask[:, :, 1:]

        score_k, score_kid = scores.topk(dim=-1, k=self.k)

        personalized_terms = news_embedding.gather(dim=-2,index=score_kid.unsqueeze(-1).expand(score_kid.size() + (news_embedding.size(-1),)))
        # weighted_ps_terms = personalized_terms * (nn.functional.softmax(score_k.masked_fill(score_k < self.threshold, 0), dim=-1).unsqueeze(-1))
        weighted_ps_terms = personalized_terms * (score_k.masked_fill(score_k < self.threshold, 0).unsqueeze(-1))

        # weighted_ps_terms.retain_grad()
        # print(weighted_ps_terms.grad, weighted_ps_terms.requires_grad)

        return weighted_ps_terms, score_kid

class BM25_Reducer(nn.Module):
    """
    topk BM25 score
    """
    def __init__(self, config):
        super().__init__()

        self.name = "bm25-reducer"

        config.term_num = config.k * config.his_size


    def forward(self, news_selection_embedding, news_embedding, user_repr, his_attn_mask):
        """
        Extract words from news text according to the overall user interest

        Args:
            news_selection_embedding: encoded word-level embedding, [batch_size, his_size, signal_length, hidden_dim]
            news_embedding: word-level news embedding, [batch_size, his_size, signal_length, hidden_dim]
            user_repr: user representation, [batch_size, 1, hidden_dim]

        Returns:
            weighted_pt: weighted embedding for personalized terms, [batch_size, his_size, k, hidden_dim]
            score_kid: index of top k terms in the text, [batch_size, his_size, k]
        """
        # strip off [CLS]
        ps_terms = news_embedding[:, :, 1:]

        return ps_terms, None