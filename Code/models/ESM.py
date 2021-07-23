import torch
import torch.nn as nn
from .base_model import BaseModel

class ESM(BaseModel):
    def __init__(self, config, encoderN, encoderU, docReducer, termFuser, interactor):
        super().__init__(config)
        self.title_size = config.title_size
        self.k = config.k

        self.encoderN = encoderN
        self.encoderU = encoderU
        self.docReducer = docReducer
        self.termFuser = termFuser
        self.interactor = interactor

        self.level = self.encoderN.level

        self.hidden_dim = encoderN.hidden_dim
        self.final_dim = interactor.final_dim

        self.learningToRank = nn.Sequential(
            nn.Linear(self.final_dim, int(self.final_dim/2)),
            nn.ReLU(),
            nn.Linear(int(self.final_dim/2),1)
        )

        self.name = '_'.join(['esm', self.encoderN.name, self.encoderU.name, self.docReducer.name, self.interactor.name])

    def clickPredictor(self, reduced_tensor):
        """ calculate batch of click probabolity

        Args:
            reduced_tensor: [batch_size, cdd_size, final_dim]

        Returns:
            score of each candidate news, [batch_size, cdd_size]
        """
        return self.learningToRank(reduced_tensor).squeeze(dim=-1)

    def _forward(self,x):
        if x['candidate_title'].size(0) != self.batch_size:
            self.batch_size = x['candidate_title'].size(0)

        cdd_news = x['candidate_title'].long().to(self.device)
        cdd_news_embedding, cdd_news_repr = self.encoderN(
            cdd_news)
        his_news = x['clicked_title'].long().to(self.device)
        his_news_embedding, his_news_repr = self.encoderN(
            his_news)

        user_repr = self.encoderU(his_news_repr)

        ps_terms, ps_term_ids = self.docReducer(his_news_embedding, user_repr)

        if self.termFuser:
            ps_terms = self.termFuser(ps_terms, ps_term_ids, his_news)
        else:
            ps_terms = ps_terms.view(self.batch_size, -1, self.level, self.hidden_dim)

        # reduced_tensor = self.interactor(torch.cat([cdd_news_repr.unsqueeze(-2), cdd_news_embedding], dim=-2), torch.cat([user_repr, ps_terms], dim=-2))
        reduced_tensor = self.interactor(cdd_news_embedding, ps_terms)

        return self.clickPredictor(reduced_tensor)

    def forward(self,x):
        """
        Decoupled function, score is unormalized click score
        """
        score = self._forward(x)

        if self.cdd_size > 1:
            prob = nn.functional.log_softmax(score, dim=1)
        else:
            prob = torch.sigmoid(score)

        return prob