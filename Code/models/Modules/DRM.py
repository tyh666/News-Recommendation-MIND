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
        self.his_size = config.his_size
        self.embedding_dim = config.embedding_dim

        config.term_num = config.k * config.his_size

        keep_k_modifier = torch.zeros(1, config.signal_length)
        keep_k_modifier[:, :self.k+1] = 1
        self.register_buffer('keep_k_modifier', keep_k_modifier, persistent=False)

        if self.diversify:
            self.newsUserAlign = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
            nn.init.xavier_normal_(self.newsUserAlign.weight)

        if config.threshold != -float('inf'):
            threshold = torch.tensor([config.threshold])
            self.register_buffer('threshold', threshold)

        if not config.no_sep_his:
            config.term_num += (self.his_size - 1)
            self.sep_embedding = nn.Parameter(torch.randn(1, 1, 1, config.embedding_dim))
            self.register_buffer('extra_sep_mask', torch.ones(1, 1, 1), persistent=False)
            nn.init.xavier_normal_(self.sep_embedding)

        if not config.no_order_embed:
            self.order_embedding = nn.Parameter(torch.randn(config.his_size, 1, config.embedding_dim))
            nn.init.xavier_normal_(self.order_embedding)


    def forward(self, news_selection_embedding, news_embedding, user_repr, news_repr, his_attn_mask, his_refined_mask):
        """
        Extract words from news text according to the overall user interest

        Args:
            news_selection_embedding: encoded word-level embedding, [batch_size, his_size, signal_length, hidden_dim]
            news_embedding: word-level news embedding, [batch_size, his_size, signal_length, hidden_dim]
            news_repr: news-level representation, [batch_size, his_size, hidden_dim]
            user_repr: user representation, [batch_size, 1, hidden_dim]

        Returns:
            ps_terms: weighted embedding for personalized terms, [batch_size, term_num, embedding_dim]
            ps_term_mask: attention mask of output terms, [batch_size, term_num]
            kid: the index of personalized terms
        """
        batch_size = news_embedding.size(0)

        # strip off [CLS]
        news_selection_embedding = news_selection_embedding[:, :, 1:]
        news_embedding = news_embedding[:, :, 1:]
        if self.diversify:
            news_user_repr = torch.cat([user_repr.expand(news_repr.size()), news_repr], dim=-1)
            selection_query = self.newsUserAlign(news_user_repr).unsqueeze(-1)
        else:
            selection_query = user_repr.unsqueeze(-1)

        # [bs, hs, sl - 1]
        scores = F.normalize(news_selection_embedding, dim=-1).matmul(F.normalize(selection_query, dim=-2)).squeeze(-1)
        # print(scores[0])
        pad_pos = ~(((his_refined_mask + self.keep_k_modifier)[:, :, 1:]).bool())
        # mask the padded term
        scores = scores.masked_fill(pad_pos, -float('inf'))

        score_k, score_kid = scores.topk(dim=-1, k=self.k)

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
        if hasattr(self, 'order_embedding'):
            ps_terms += self.order_embedding

        if hasattr(self, 'sep_embedding'):
            ps_terms = torch.cat([ps_terms, self.sep_embedding.expand(batch_size, self.his_size, 1, self.embedding_dim)], dim=-2).view(batch_size, -1, self.embedding_dim)[:, :-1]
            ps_term_mask = torch.cat([ps_term_mask, self.extra_sep_mask.expand(batch_size, self.his_size, 1)], dim=-1).view(batch_size, -1)[:, :-1]
        else:
            ps_terms = ps_terms.view(batch_size, -1, self.embedding_dim)
            ps_term_mask = ps_term_mask.view(batch_size, -1)

        return ps_terms, ps_term_mask, score_kid


class Slicing_Reducer(nn.Module):
    """
    truncation
    """
    def __init__(self, config):
        super().__init__()

        self.name = "slicing"
        self.k = config.k
        self.his_size = config.his_size
        self.embedding_dim = config.embedding_dim

        config.term_num = config.k * config.his_size

        if not config.no_sep_his:
            config.term_num += (self.his_size - 1)
            self.sep_embedding = nn.Parameter(torch.randn(1, 1, 1, config.embedding_dim))
            self.register_buffer('extra_sep_mask', torch.ones(1, 1, 1), persistent=False)
            nn.init.xavier_normal_(self.sep_embedding)

        if not config.no_order_embed:
            self.order_embedding = nn.Parameter(torch.randn(config.his_size, 1, config.embedding_dim))
            nn.init.xavier_normal_(self.order_embedding)

        self.register_buffer('kid', torch.arange(config.k).unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, news_selection_embedding, news_embedding, user_repr, news_repr, his_attn_mask, his_refined_mask=None):
        """
        Extract words from news text according to the overall user interest

        Args:
            news_selection_embedding: encoded word-level embedding, [batch_size, his_size, signal_length, hidden_dim]
            news_embedding: word-level news embedding, [batch_size, his_size, signal_length, hidden_dim]
            user_repr: user representation, [batch_size, 1, hidden_dim]

        Returns:
            ps_terms: weighted embedding for personalized terms, [batch_size, his_size, k, hidden_dim]
            ps_term_mask: attention mask of output terms, [batch_size, his_size, k]
            kid: the index of personalized terms
        """
        # strip off [CLS]
        ps_terms = news_embedding[:, :, 1:]
        ps_term_mask = his_attn_mask[:, :, 1:]

        batch_size = ps_terms.size(0)

        if hasattr(self, 'order_embedding'):
            ps_terms += self.order_embedding

        if hasattr(self, 'sep_embedding'):
            ps_terms = torch.cat([ps_terms, self.sep_embedding.expand(batch_size, self.his_size, 1, self.embedding_dim)], dim=-2).view(batch_size, -1, self.embedding_dim)[:, :-1]
            ps_term_mask = torch.cat([ps_term_mask, self.extra_sep_mask.expand(batch_size, self.his_size, 1)], dim=-1).view(batch_size, -1)[:, :-1]
        else:
            ps_terms = ps_terms.view(batch_size, -1, self.embedding_dim)
            ps_term_mask = ps_term_mask.view(batch_size, -1)

        return ps_terms, ps_term_mask, self.kid.expand(batch_size, self.his_size, self.k)