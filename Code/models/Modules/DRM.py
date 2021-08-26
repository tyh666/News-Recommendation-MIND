import torch
import torch.nn as nn
import torch.nn.functional as F

class Matching_Reducer(nn.Module):
    """
    basic document reducer: topk of each historical article
    """
    def __init__(self, config):
        super().__init__()

        self.name = "matching"

        self.k = config.k
        self.diversify = config.diversify

        config.term_num = config.k * config.his_size

        if self.diversify:
            self.newsUserAlign = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
            nn.init.xavier_normal_(self.newsUserAlign.weight)

        if config.threshold != -float('inf'):
            threshold = torch.tensor([config.threshold])
            self.register_buffer('threshold', threshold)

    def forward(self, news_selection_embedding, news_embedding, user_repr, news_repr, his_attn_mask, his_attn_mask_k):
        """
        Extract words from news text according to the overall user interest

        Args:
            news_selection_embedding: encoded word-level embedding, [batch_size, his_size, signal_length, hidden_dim]
            news_embedding: word-level news embedding, [batch_size, his_size, signal_length, hidden_dim]
            news_repr: news-level representation, [batch_size, his_size, hidden_dim]
            user_repr: user representation, [batch_size, 1, hidden_dim]

        Returns:
            ps_terms: weighted embedding for personalized terms, [batch_size, his_size, k, hidden_dim]
            ps_term_mask: attention mask of output terms, [batch_size, his_size, k]
        """
        # strip off [CLS]
        news_selection_embedding = news_selection_embedding[:, :, 1:]
        news_embedding = news_embedding[:, :, 1:]
        if self.diversify:
            news_user_repr = torch.cat([user_repr.expand(news_repr.size()), news_repr], dim=-1)
            selection_query = self.newsUserAlign(news_user_repr).unsqueeze(-1)
        else:
            selection_query = user_repr.expand(news_repr.size()).unsqueeze(-1)

        # [bs, hs, sl - 1]
        scores = F.normalize(news_selection_embedding, dim=-1).matmul(F.normalize(selection_query, dim=-1)).squeeze(-1)
        # mask the padded term
        scores = scores.masked_fill(~his_attn_mask_k[:, :, 1:], -float('inf'))

        score_k, score_kid = scores.topk(dim=-1, k=self.k, sorted=False)

        ps_terms = news_embedding.gather(dim=-2,index=score_kid.unsqueeze(-1).expand(score_kid.size() + (news_embedding.size(-1),)))
        # [bs, hs, k]
        ps_term_mask = his_attn_mask[:, :, 1:].gather(dim=-1, index=score_kid)

        if hasattr(self, 'threshold'):
            mask_pos = score_k < self.threshold
            # ps_terms = personalized_terms * (nn.functional.softmax(score_k.masked_fill(score_k < self.threshold, 0), dim=-1).unsqueeze(-1))
            ps_terms = ps_terms * (score_k.masked_fill(mask_pos, 0).unsqueeze(-1))
            ps_term_mask = ps_term_mask * (~mask_pos)

        else:
            ps_terms = ps_terms * (F.softmax(score_k, dim=-1).unsqueeze(-1))
            # ps_terms = ps_terms * (score_k.unsqueeze(-1))

        return ps_terms, ps_term_mask, score_kid


class BM25_Reducer(nn.Module):
    """
    topk BM25 score
    """
    def __init__(self, config):
        super().__init__()

        self.name = "bm25"

        config.term_num = config.k * config.his_size


    def forward(self, news_selection_embedding, news_embedding, user_repr, news_repr, his_attn_mask, his_attn_mask_k=None):
        """
        Extract words from news text according to the overall user interest

        Args:
            news_selection_embedding: encoded word-level embedding, [batch_size, his_size, signal_length, hidden_dim]
            news_embedding: word-level news embedding, [batch_size, his_size, signal_length, hidden_dim]
            user_repr: user representation, [batch_size, 1, hidden_dim]

        Returns:
            ps_terms: weighted embedding for personalized terms, [batch_size, his_size, k, hidden_dim]
            ps_term_mask: attention mask of output terms, [batch_size, his_size, k]
        """
        # strip off [CLS]
        ps_terms = news_embedding[:, :, 1:]
        ps_term_mask = his_attn_mask[:, :, 1:]

        return ps_terms, ps_term_mask


class BOW_Reducer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = "bow"
        self.k = config.k
        self.diversify = config.diversify

        config.term_num = config.k * config.his_size

        if self.diversify:
            self.newsUserAlign = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
            nn.init.xavier_normal_(self.newsUserAlign.weight)

        if config.threshold != -float('inf'):
            threshold = torch.tensor([config.threshold])
            self.register_buffer('threshold', threshold)

    def forward(self, news_selection_embedding, news_embedding, user_repr, news_repr, his_attn_mask):
        """
        Extract words from news text according to the overall user interest

        Args:
            news_selection_embedding: encoded word-level embedding, [batch_size, his_size, signal_length, hidden_dim]
            news_embedding: word-level news embedding, [batch_size, his_size, signal_length, hidden_dim]
            news_repr: news-level representation, [batch_size, his_size, hidden_dim]
            user_repr: user representation, [batch_size, 1, hidden_dim]

        Returns:
            ps_terms: weighted embedding for personalized terms, [batch_size, his_size, k, hidden_dim]
            ps_term_mask: attention mask of output terms, [batch_size, his_size, k]
        """
        # strip off [CLS]
        news_selection_embedding = news_selection_embedding[:, :, 1:]
        news_embedding = news_embedding[:, :, 1:]
        if self.diversify:
            news_user_repr = torch.cat([user_repr.expand(news_repr.size()), news_repr], dim=-1)
            selection_query = self.newsUserAlign(news_user_repr).unsqueeze(-1)
        else:
            selection_query = user_repr.expand(news_repr.size()).unsqueeze(-1)

        # [bs, hs, sl - 1]
        scores = F.normalize(news_selection_embedding, dim=-1).matmul(F.normalize(selection_query, dim=-1)).squeeze(-1)
        # mask the padded term

        score_k, score_kid = scores.topk(dim=-1, k=self.k, sorted=False)

        ps_terms = news_embedding.gather(dim=-2,index=score_kid.unsqueeze(-1).expand(score_kid.size() + (news_embedding.size(-1),)))
        # [bs, hs, k]
        ps_term_mask = his_attn_mask[:, :, 1:].gather(dim=-1, index=score_kid)

        if hasattr(self, 'threshold'):
            mask_pos = score_k < self.threshold
            # ps_terms = personalized_terms * (nn.functional.softmax(score_k.masked_fill(score_k < self.threshold, 0), dim=-1).unsqueeze(-1))
            ps_terms = ps_terms * (score_k.masked_fill(mask_pos, 0).unsqueeze(-1))
            ps_term_mask = ps_term_mask * (~mask_pos)

        else:
            ps_terms = ps_terms * (F.softmax(score_k, dim=-1).unsqueeze(-1))
            # ps_terms = ps_terms * (score_k.unsqueeze(-1))

        return ps_terms, ps_term_mask, score_kid