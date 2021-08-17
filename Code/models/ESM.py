import torch
import torch.nn as nn

class ESM(nn.Module):
    def __init__(self, config, embedding, encoderN, encoderU, docReducer, termFuser, interactor):
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
        self.docReducer = docReducer
        self.termFuser = termFuser
        self.interactor = interactor

        self.hidden_dim = encoderN.hidden_dim
        self.final_dim = interactor.final_dim

        self.learningToRank = nn.Sequential(
            nn.Linear(self.final_dim + 1, int(self.final_dim/2)),
            nn.ReLU(),
            nn.Linear(int(self.final_dim/2),1)
        )

        self.name = '__'.join(['esm', self.encoderN.name, self.encoderU.name, self.docReducer.name, self.interactor.name])
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
        score_coarse = cdd_news_repr.matmul(user_repr.transpose(-2,-1))
        score = torch.cat([reduced_tensor, score_coarse], dim=-1)

        return self.learningToRank(score).squeeze(dim=-1)

    def _forward(self,x):
        if x["cdd_encoded_index"].size(0) != self.batch_size:
            self.batch_size = x["cdd_encoded_index"].size(0)

        cdd_news = x["cdd_encoded_index"].long().to(self.device)
        cdd_news_embedding = self.embedding(cdd_news)
        _, cdd_news_repr = self.encoderN(
            cdd_news_embedding
        )
        his_news = x["his_encoded_index"].long().to(self.device)
        his_news_embedding = self.embedding(his_news)
        his_news_encoded_embedding, his_news_repr = self.encoderN(
            his_news_embedding
        )

        user_repr = self.encoderU(his_news_repr)

        ps_terms, ps_term_ids = self.docReducer(his_news_encoded_embedding, his_news_embedding, user_repr)

        # if self.termFuser:
        #     ps_terms = self.termFuser(ps_terms, ps_term_ids, his_news)

        # reduced_tensor = self.interactor(torch.cat([cdd_news_repr.unsqueeze(-2), cdd_news_embedding], dim=-2), torch.cat([user_repr, ps_terms], dim=-2))

        reduced_tensor = self.interactor(cdd_news_embedding, ps_terms, x["cdd_attn_mask"].to(self.device))

        return self.clickPredictor(reduced_tensor, cdd_news_repr, user_repr)

    def forward(self,x):
        """
        Decoupled function, score is unormalized click score
        """
        score = self._forward(x)

        if self.training:
            prob = nn.functional.log_softmax(score, dim=1)
        else:
            prob = torch.sigmoid(score)

        return prob