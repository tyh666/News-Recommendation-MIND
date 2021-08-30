# Personalized terms
import torch
import torch.nn as nn
import torch.nn.functional as F

class ESM(nn.Module):
    def __init__(self, config, embedding, encoderN, encoderU, reducer, fuser, ranker):
        super().__init__()

        self.scale = config.scale
        self.cdd_size = config.cdd_size
        self.batch_size = config.batch_size
        self.his_size = config.his_size
        self.signal_length = config.signal_length
        self.device = config.device

        self.k = config.k

        self.embedding = embedding
        self.encoderN = encoderN
        self.encoderU = encoderU
        self.reducer = reducer
        self.fuser = fuser
        self.ranker = ranker

        self.final_dim = ranker.final_dim

        self.learningToRank = nn.Sequential(
            nn.Linear(self.final_dim + 1, int(self.final_dim/2)),
            nn.ReLU(),
            nn.Linear(int(self.final_dim/2),1)
        )

        self.word_level = config.word_level
        if self.word_level:
            self.register_buffer('cdd_dest', torch.zeros((self.batch_size, config.impr_size, self.signal_length * self.signal_length)), persistent=False)
            self.register_buffer('his_dest', torch.zeros((self.batch_size, self.his_size, self.signal_length * self.signal_length)), persistent=False)

        self.name = '__'.join(['esm', self.encoderN.name, self.encoderU.name, self.reducer.name, self.ranker.name])
        config.name = self.name

    def clickPredictor(self, reduced_tensor, cdd_news_repr, user_repr):
        """ calculate batch of click probabolity

        Args:
            reduced_tensor: [batch_size, cdd_size, final_dim]
            cdd_news_repr: news-level representation, [batch_size, cdd_size, hidden_dim]
            user_repr: user representation, [batch_size, 1, hidden_dim]

        Returns:
            score of each candidate news, [batch_size, cdd_size]
        """

        # print(user_repr.mean(), cdd_news_repr.mean(), user_repr.max(), cdd_news_repr.max(), user_repr.sum(), cdd_news_repr.sum())
        score_coarse = cdd_news_repr.matmul(user_repr.transpose(-2,-1))
        score = torch.cat([reduced_tensor, score_coarse], dim=-1)

        return self.learningToRank(score).squeeze(dim=-1)

    def _forward(self,x):
        if self.word_level:
            batch_size = x['cdd_subword_index'].size(0)
            cdd_size = x['cdd_subword_index'].size(1)

            if self.training:
                if batch_size != self.batch_size:
                    cdd_dest = self.cdd_dest[:batch_size, :cdd_size]
                    his_dest = self.his_dest[:batch_size]
                else:
                    cdd_dest = self.cdd_dest[:, :cdd_size]
                    his_dest = self.his_dest

            # batch_size always equals 1 when evaluating
            else:
                cdd_dest = self.cdd_dest[[0], :cdd_size]
                his_dest = self.his_dest[[0]]

            cdd_subword_index = x['cdd_subword_index'].to(self.device)
            cdd_subword_index = cdd_subword_index[:, :, :, 0] * self.signal_length + cdd_subword_index[:, :, :, 1]
            his_subword_index = x['his_subword_index'].to(self.device)
            his_subword_index = his_subword_index[:, :, :, 0] * self.signal_length + his_subword_index[:, :, :, 1]

            if self.training:
                cdd_subword_prefix = cdd_dest.scatter(dim=-1, index=cdd_subword_index, value=1) * x["cdd_mask"].to(self.device)
            else:
                cdd_subword_prefix = cdd_dest.scatter(dim=-1, index=cdd_subword_index, value=1)
            cdd_subword_prefix = cdd_subword_prefix.view(batch_size, cdd_size, self.signal_length, self.signal_length)

            his_subword_prefix = his_dest.scatter(dim=-1, index=his_subword_index, value=1) * x["his_mask"].to(self.device)
            his_subword_prefix = his_subword_prefix.view(batch_size, self.his_size, self.signal_length, self.signal_length)

            cdd_subword_prefix = F.normalize(cdd_subword_prefix, p=1, dim=-1)
            his_subword_prefix = F.normalize(his_subword_prefix, p=1, dim=-1)

            his_attn_mask = his_subword_prefix.matmul(x["his_attn_mask"].to(self.device).float().unsqueeze(-1)).squeeze(-1)
            his_reduced_mask = his_subword_prefix.matmul(x["his_reduced_mask"].to(self.device).float().unsqueeze(-1)).squeeze(-1)

            cdd_attn_mask = cdd_subword_prefix.matmul(x['cdd_attn_mask'].to(self.device).float().unsqueeze(-1)).squeeze(-1)

        else:
            cdd_subword_prefix = None
            his_subword_prefix = None
            his_attn_mask = x["his_attn_mask"].to(self.device)
            his_reduced_mask = x["his_reduced_mask"].to(self.device)
            cdd_attn_mask = x['cdd_attn_mask'].to(self.device)


        cdd_news = x["cdd_encoded_index"].long().to(self.device)
        cdd_news_embedding = self.embedding(cdd_news, cdd_subword_prefix)
        _, cdd_news_repr = self.encoderN(
            cdd_news_embedding
        )

        if self.reducer.name == 'matching':
            his_news = x["his_encoded_index"].long().to(self.device)
            his_news_embedding = self.embedding(his_news, his_subword_prefix)
            his_news_encoded_embedding, his_news_repr = self.encoderN(
                his_news_embedding
            )

            user_repr = self.encoderU(his_news_repr)

            ps_terms, ps_term_mask, kid = self.reducer(his_news_encoded_embedding, his_news_embedding, user_repr, his_news_repr, his_attn_mask, his_reduced_mask)

        elif self.reducer.name == 'bow':
            his_news = x["his_reduced_index"].long().to(self.device)
            his_news_embedding = self.embedding(his_news, bow=True)
            his_news_encoded_embedding, his_news_repr = self.encoderN(
                his_news_embedding
            )
            user_repr = self.encoderU(his_news_repr)

            ps_terms, ps_term_mask, kid = self.reducer(his_news_encoded_embedding, his_news_embedding, user_repr, his_news_repr, x["his_attn_mask"].to(self.device), x["his_reduced_mask"].to(self.device).bool())

        elif self.reducer.name == 'bm25':
            kid = None
            his_news = x["his_reduced_index"].long().to(self.device)
            his_news_embedding = self.embedding(his_news)
            his_news_encoded_embedding, his_news_repr = self.encoderN(
                his_news_embedding
            )

            user_repr = None

            ps_terms, ps_term_mask = self.reducer(his_news_encoded_embedding, his_news_embedding, user_repr, his_news_repr, x["his_reduced_mask"].to(self.device))


        if self.fuser:
            ps_terms, ps_term_mask = self.fuser(ps_terms, ps_term_mask)

        # reduced_tensor = self.ranker(torch.cat([cdd_news_repr.unsqueeze(-2), cdd_news_embedding], dim=-2), torch.cat([user_repr, ps_terms], dim=-2))

        reduced_tensor = self.ranker(cdd_news_embedding, ps_terms, cdd_attn_mask, ps_term_mask)
        # reduced_tensor = self.ranker(cdd_news_embedding, ps_terms, x['cdd_attn_mask'].to(self.device), ps_term_mask)

        return self.clickPredictor(reduced_tensor, cdd_news_repr, user_repr), kid

    def forward(self,x):
        """
        Decoupled function, score is unormalized click score
        """
        score, kid = self._forward(x)

        if self.training:
            prob = nn.functional.log_softmax(score, dim=1)
        else:
            prob = torch.sigmoid(score)

        return (prob, kid)