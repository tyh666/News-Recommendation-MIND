# Personalized terms
import torch
import torch.nn as nn

class ESM(nn.Module):
    def __init__(self, config, embedding, encoderN, encoderU, reducer, fuser, ranker):
        super().__init__()

        self.scale = config.scale
        self.cdd_size = config.cdd_size
        self.batch_size = config.batch_size
        self.his_size = config.his_size
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
        cdd_news = x["cdd_encoded_index"].long().to(self.device)
        cdd_news_embedding = self.embedding(cdd_news)
        _, cdd_news_repr = self.encoderN(
            cdd_news_embedding
        )

        if self.reducer.name == 'matching':
            his_news = x["his_encoded_index"].long().to(self.device)
            his_news_embedding = self.embedding(his_news)
            his_news_encoded_embedding, his_news_repr = self.encoderN(
                his_news_embedding
            )

            user_repr = self.encoderU(his_news_repr)

            ps_terms, ps_term_mask, kid = self.reducer(his_news_encoded_embedding, his_news_embedding, user_repr, his_news_repr, x["his_attn_mask"].to(self.device), x["his_reduced_mask"].to(self.device).bool())

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

        reduced_tensor = self.ranker(cdd_news_embedding, ps_terms, x["cdd_attn_mask"].to(self.device), ps_term_mask)

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