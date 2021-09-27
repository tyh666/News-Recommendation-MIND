# Personalized terms
import torch
import torch.nn as nn
import torch.nn.functional as F

class ESM(nn.Module):
    def __init__(self, manager, embedding, encoderN, encoderU, reducer, ranker):
        super().__init__()

        self.scale = manager.scale
        self.cdd_size = manager.cdd_size
        self.batch_size = manager.batch_size
        self.his_size = manager.his_size
        self.signal_length = manager.signal_length
        self.device = manager.device

        self.k = manager.k

        self.embedding = embedding
        self.encoderN = encoderN
        self.encoderU = encoderU
        self.reducer = reducer
        self.ranker = ranker

        self.final_dim = ranker.hidden_dim

        self.learningToRank = nn.Sequential(
            nn.Linear(self.final_dim, int(self.final_dim/2)),
            nn.Tanh(),
            nn.Linear(int(self.final_dim/2),1)
        )
        nn.init.xavier_normal_(self.learningToRank[0].weight)
        nn.init.xavier_normal_(self.learningToRank[2].weight)

        self.granularity = manager.granularity
        if self.granularity != 'token':
            self.register_buffer('cdd_dest', torch.zeros((self.batch_size, manager.impr_size, manager.signal_length * manager.signal_length)), persistent=False)
            if manager.reducer in ["bm25", "entity", "first"]:
                self.register_buffer('his_dest', torch.zeros((self.batch_size, self.his_size, (manager.k + 1) * (manager.k + 1))), persistent=False)
            else:
                self.register_buffer('his_dest', torch.zeros((self.batch_size, self.his_size, manager.signal_length * manager.signal_length)), persistent=False)

        manager.name = '__'.join(['esm', manager.embedding, manager.encoderN, manager.encoderU, manager.reducer, manager.ranker, manager.granularity, "full" if manager.full_attn else "partial"])


    def clickPredictor(self, reduced_tensor):
        """ calculate batch of click probabolity

        Args:
            reduced_tensor: [batch_size, cdd_size, final_dim]
            cdd_news_repr: news-level representation, [batch_size, cdd_size, hidden_dim]
            user_repr: user representation, [batch_size, 1, hidden_dim]

        Returns:
            score of each candidate news, [batch_size, cdd_size]
        """

        # print(user_repr.mean(), cdd_news_repr.mean(), user_repr.max(), cdd_news_repr.max(), user_repr.sum(), cdd_news_repr.sum())
        # score_coarse = cdd_news_repr.matmul(user_repr.transpose(-2,-1))
        # score = torch.cat([reduced_tensor, score_coarse], dim=-1)
        score = self.learningToRank(reduced_tensor).squeeze(dim=-1)

        return score

    def _forward(self,x):
        if self.granularity != 'token':
            batch_size = x['cdd_subword_index'].size(0)
            cdd_size = x['cdd_subword_index'].size(1)

            if self.training:
                cdd_dest = self.cdd_dest[:batch_size, :cdd_size]
                his_dest = self.his_dest[:batch_size]

            # batch_size always equals 1 when evaluating
            else:
                cdd_dest = self.cdd_dest[[0], :cdd_size]
                his_dest = self.his_dest[[0]]

            cdd_subword_index = x['cdd_subword_index'].to(self.device)
            his_subword_index = x['his_subword_index'].to(self.device)
            his_signal_length = his_subword_index.size(-2)
            cdd_subword_index = cdd_subword_index[:, :, :, 0] * self.signal_length + cdd_subword_index[:, :, :, 1]
            his_subword_index = his_subword_index[:, :, :, 0] * his_signal_length + his_subword_index[:, :, :, 1]

            if self.training:
                # * cdd_mask to filter out padded cdd news
                cdd_subword_prefix = cdd_dest.scatter(dim=-1, index=cdd_subword_index, value=1) * x["cdd_mask"].to(self.device)
            else:
                cdd_subword_prefix = cdd_dest.scatter(dim=-1, index=cdd_subword_index, value=1)
            cdd_subword_prefix = cdd_subword_prefix.view(batch_size, cdd_size, self.signal_length, self.signal_length)

            his_subword_prefix = his_dest.scatter(dim=-1, index=his_subword_index, value=1) * x["his_mask"].to(self.device)
            his_subword_prefix = his_subword_prefix.view(batch_size, self.his_size, his_signal_length, his_signal_length)

            if self.granularity == 'avg':
                # average subword embeddings as the word embedding
                cdd_subword_prefix = F.normalize(cdd_subword_prefix, p=1, dim=-1)
                his_subword_prefix = F.normalize(his_subword_prefix, p=1, dim=-1)

            cdd_attn_mask = cdd_subword_prefix.matmul(x['cdd_attn_mask'].to(self.device).float().unsqueeze(-1)).squeeze(-1)
            his_attn_mask = his_subword_prefix.matmul(x["his_attn_mask"].to(self.device).float().unsqueeze(-1)).squeeze(-1)
            his_refined_mask = None
            if 'his_refined_mask' in x:
                his_refined_mask = his_subword_prefix.matmul(x["his_refined_mask"].to(self.device).float().unsqueeze(-1)).squeeze(-1)

        else:
            cdd_subword_prefix = None
            his_subword_prefix = None
            cdd_attn_mask = x['cdd_attn_mask'].to(self.device)
            his_attn_mask = x["his_attn_mask"].to(self.device)
            his_refined_mask = None
            if 'his_refined_mask' in x:
                his_refined_mask = x["his_refined_mask"].to(self.device)


        cdd_news = x["cdd_encoded_index"].long().to(self.device)
        cdd_news_embedding = self.embedding(cdd_news, cdd_subword_prefix)

        his_news = x["his_encoded_index"].long().to(self.device)
        his_news_embedding = self.embedding(his_news, his_subword_prefix)
        his_news_encoded_embedding, his_news_repr = self.encoderN(his_news_embedding, his_attn_mask)

        user_repr = self.encoderU(his_news_repr)

        ps_terms, ps_term_mask, kid = self.reducer(his_news_encoded_embedding, his_news_embedding, user_repr, his_news_repr, his_attn_mask, his_refined_mask)

        reduced_tensor = self.ranker(cdd_news_embedding, ps_terms, cdd_attn_mask, ps_term_mask)

        return self.clickPredictor(reduced_tensor), kid

    def forward(self,x):
        """
        Decoupled function, score is unormalized click score
        """
        score, kid = self._forward(x)

        if self.training:
            prob = nn.functional.log_softmax(score, dim=1)
        else:
            prob = torch.sigmoid(score)

        return prob, kid