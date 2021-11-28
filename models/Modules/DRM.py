import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Matching_Reducer(nn.Module):
    """
    select top k terms from each historical news with max cosine similarity

    1. keep the first K terms unmasked
    2. add order embedding to terms from different historical news
    3. insert [SEP] token to separate terms from different news if called
    """
    def __init__(self, manager):
        super().__init__()
        self.k = manager.k
        if manager.mode == "inspect":
            self.k = 10

        self.his_size = manager.his_size
        self.embedding_dim = manager.embedding_dim

        self.diversify = manager.diversify
        self.sep_his = manager.sep_his
        # if aggregator is enabled, do not flatten the personalized terms
        self.flatten = (manager.aggregator is None)

        manager.term_num = manager.k * manager.his_size

        # strip [CLS]
        keep_k_modifier = torch.zeros(1, manager.signal_length - 1)
        keep_k_modifier[:, :self.k] = 1
        self.register_buffer('keep_k_modifier', keep_k_modifier, persistent=False)

        if self.diversify:
            self.newsUserAlign = nn.Linear(manager.hidden_dim * 2, manager.hidden_dim)
            nn.init.xavier_normal_(self.newsUserAlign.weight)

        if self.sep_his:
            manager.term_num += (self.his_size - 1)
            self.sep_embedding = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
            self.register_buffer('extra_sep_mask', torch.ones(1, 1, 1), persistent=False)

        if manager.segment_embed:
            self.segment_embedding = nn.Parameter(torch.randn(manager.his_size, 1, manager.embedding_dim))
            nn.init.xavier_normal_(self.segment_embedding)


    def forward(self, news_selection_embedding, news_embedding, user_repr, news_repr, his_attn_mask, his_refined_mask):
        """
        Extract words from news text according to the overall user interest

        Args:
            news_selection_embedding: encoded word-level embedding, [batch_size, his_size, signal_length, hidden_dim]
            news_embedding: word-level news embedding, [batch_size, his_size, signal_length, hidden_dim]
            news_repr: news-level representation, [batch_size, his_size, hidden_dim]
            user_repr: user representation, [batch_size, 1, hidden_dim]
            his_refined_mask: dedupicated attention mask, [batch_size, his_size, signal_length]
        Returns:
            ps_terms: weighted embedding for personalized terms, [batch_size, term_num, embedding_dim]
            ps_term_mask: attention mask of output terms, [batch_size, term_num]
            kid: the index of personalized terms
        """
        batch_size = news_embedding.size(0)

        if self.diversify:
            news_user_repr = torch.cat([user_repr.expand(news_repr.size()), news_repr], dim=-1)
            selection_query = self.newsUserAlign(news_user_repr).unsqueeze(-1)
        else:
            selection_query = user_repr.unsqueeze(-1)

        news_selection_embedding = news_selection_embedding[:, :, 1:]

        news_embedding_text = news_embedding[:, :, 1:]
        his_attn_mask = his_attn_mask[:, :, 1:]

        # [bs, hs, sl - 1]
        scores = F.normalize(news_selection_embedding, dim=-1).matmul(F.normalize(selection_query, dim=-2)).squeeze(-1)
        # scores = news_selection_embedding.matmul(selection_query).squeeze(-1)/math.sqrt(selection_query.size(-1))
        pad_pos = ~((his_refined_mask[:, :, 1:] + self.keep_k_modifier).bool())

        # mask the padded term
        scores = scores.masked_fill(pad_pos, -float('inf'))

        score_k, score_kid = scores.topk(dim=-1, k=self.k, sorted=True)

        ps_terms = news_embedding_text.gather(dim=-2,index=score_kid.unsqueeze(-1).expand(*score_kid.size(), news_embedding_text.size(-1)))
        # [bs, hs, k]
        ps_term_mask = his_attn_mask.gather(dim=-1, index=score_kid)

        if hasattr(self, 'threshold'):
            mask_pos = score_k < self.threshold
            # ps_terms = personalized_terms * (nn.functional.softmax(score_k.masked_fill(score_k < self.threshold, 0), dim=-1).unsqueeze(-1))
            ps_terms = ps_terms * (F.softmax(score_k.masked_fill(mask_pos, 0), dim=-1).unsqueeze(-1))
            ps_term_mask = ps_term_mask * (~mask_pos)
        else:
            ps_terms = ps_terms * (F.softmax(score_k, dim=-1).unsqueeze(-1))

        if hasattr(self, 'segment_embedding'):
            ps_terms += self.segment_embedding

        # flatten the selected terms into one dimension
        if self.flatten:
            # separate historical news only practical when squeeze=True
            if self.sep_his:
                # [bs, hs, ed]
                sep_embedding = self.sep_embedding.expand(batch_size, self.his_size, 1, self.embedding_dim)
                # add extra [SEP] token to separate terms from different history news, slice to -1 to strip off the last [SEP]
                ps_terms = torch.cat([ps_terms, sep_embedding], dim=-2).view(batch_size, -1, self.embedding_dim)[:, :-1]
                ps_term_mask = torch.cat([ps_term_mask, self.extra_sep_mask.expand(batch_size, self.his_size, 1)], dim=-1).view(batch_size, -1)[:, :-1]

            else:
                # [bs, 1, ed]
                ps_terms = ps_terms.reshape(batch_size, -1, self.embedding_dim)
                ps_term_mask = ps_term_mask.reshape(batch_size, -1)

        return ps_terms, ps_term_mask, score_kid


class Identical_Reducer(nn.Module):
    """
    do nothing
    """
    def __init__(self, manager):
        super().__init__()

        self.k = manager.k
        self.his_size = manager.his_size
        self.embedding_dim = manager.embedding_dim

        manager.term_num = manager.k * manager.his_size

        self.sep_his = manager.sep_his
        # if aggregator is enabled, do not flatten the personalized terms
        self.flatten = (manager.aggregator is None)

        if self.sep_his:
            manager.term_num += (self.his_size - 1)
            self.sep_embedding = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
            self.register_buffer('extra_sep_mask', torch.ones(1, 1, 1), persistent=False)

        if manager.segment_embed:
            self.segment_embedding = nn.Parameter(torch.randn(manager.his_size, 1, manager.embedding_dim))
            nn.init.xavier_normal_(self.segment_embedding)

        self.register_buffer('kid', torch.arange(manager.k).unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, news_selection_embedding, news_embedding, user_repr, news_repr, his_attn_mask, his_refined_mask=None, squeeze=True):
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
        # strip off [CLS] and [SEP]
        ps_terms = news_embedding[:, :, 1:]
        ps_term_mask = his_attn_mask[:, :, 1:]

        batch_size = ps_terms.size(0)

        if hasattr(self, 'order_embedding'):
            ps_terms += self.segment_embedding

        # flatten the selected terms into one dimension
        if self.flatten:
            # separate historical news only practical when squeeze=True
            if self.sep_his:
                # [bs, hs, ed]
                sep_embedding = self.sep_embedding.expand(batch_size, self.his_size, 1, self.embedding_dim)
                # add extra [SEP] token to separate terms from different history news, slice to -1 to strip off the last [SEP]
                ps_terms = torch.cat([ps_terms, sep_embedding], dim=-2).view(batch_size, -1, self.embedding_dim)[:, :-1]
                ps_term_mask = torch.cat([ps_term_mask, self.extra_sep_mask.expand(batch_size, self.his_size, 1)], dim=-1).view(batch_size, -1)[:, :-1]

            else:
                # [bs, 1, ed]
                ps_terms = ps_terms.reshape(batch_size, -1, self.embedding_dim)
                ps_term_mask = ps_term_mask.reshape(batch_size, -1)

        return ps_terms, ps_term_mask, self.kid.expand(batch_size, self.his_size, self.k)


class Truncating_Reducer(nn.Module):
    """
    truncation
    """
    def __init__(self, manager):
        super().__init__()

        self.k = manager.k
        self.his_size = manager.his_size
        self.embedding_dim = manager.embedding_dim

        self.sep_his = manager.sep_his
        self.max_length = manager.get_max_length_for_truncating()
        # if aggregator is enabled, do not flatten the personalized terms
        self.flatten = (manager.aggregator is None)

        manager.term_num = manager.k * manager.his_size

        if self.sep_his:
            manager.term_num += (self.his_size - 1)
            self.sep_embedding = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
            self.register_buffer('extra_sep_mask', torch.ones(1, 1, 1), persistent=False)

        if manager.segment_embed:
            self.segment_embedding = nn.Parameter(torch.randn(manager.his_size, 1, manager.embedding_dim))
            nn.init.xavier_normal_(self.segment_embedding)

    def forward(self, news_selection_embedding, news_embedding, user_repr, news_repr, his_attn_mask, his_refined_mask=None, squeeze=True):
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
        assert squeeze == True, "squeeze must be True for XFormer models"
        # strip off [CLS] and [SEP]
        ps_terms = news_embedding[:, :, 1:]
        ps_term_mask = his_attn_mask[:, :, 1:]

        batch_size = ps_terms.size(0)

        ps_terms = ps_terms.reshape(batch_size, -1, self.embedding_dim)[:, :self.max_length]
        ps_term_mask = ps_term_mask.reshape(batch_size, -1)[:, :self.max_length]

        if hasattr(self, 'order_embedding'):
            ps_terms += self.segment_embedding

        if self.sep_his:
            # [bs, hs, ed]
            sep_embedding = self.sep_embedding.expand(batch_size, self.his_size, 1, self.embedding_dim)
            # add extra [SEP] token to separate terms from different history news, slice to -1 to strip off the last [SEP]
            ps_terms = torch.cat([ps_terms, sep_embedding], dim=-2).view(batch_size, -1, self.embedding_dim)[:, :-1]
            ps_term_mask = torch.cat([ps_term_mask, self.extra_sep_mask.expand(batch_size, self.his_size, 1)], dim=-1).view(batch_size, -1)[:, :-1]

        else:
            # [bs, 1, ed]
            ps_terms = ps_terms.reshape(batch_size, -1, self.embedding_dim)
            ps_term_mask = ps_term_mask.reshape(batch_size, -1)

        return ps_terms, ps_term_mask, None