# Two tower baseline
import torch
import torch.nn as nn

from .Encoders.BERT import BERT_Encoder
from .Modules.DRM import Matching_Reducer

class TTMS(nn.Module):
    def __init__(self, config, embedding, encoderN, encoderU, reducer, aggregator=None):
        super().__init__()

        self.scale = config.scale
        self.cdd_size = config.cdd_size
        self.batch_size = config.batch_size
        self.his_size = config.his_size
        self.device = config.device

        self.embedding = embedding
        self.encoderN = encoderN
        self.encoderU = encoderU

        self.reducer = reducer
        self.bert = BERT_Encoder(config)

        self.aggregator = aggregator

        if not aggregator:
            self.userProject = nn.Sequential(
                nn.Linear(self.bert.hidden_dim, self.bert.hidden_dim),
                nn.Tanh()
            )

        self.name = '__'.join(['ttms', self.encoderN.name, self.encoderU.name, self.reducer.name])
        config.name = self.name

    def clickPredictor(self, cdd_news_repr, user_repr):
        """ calculate batch of click probabolity

        Args:
            cdd_news_repr: news-level representation, [batch_size, cdd_size, hidden_dim]
            user_repr: user representation, [batch_size, 1, hidden_dim]

        Returns:
            score of each candidate news, [batch_size, cdd_size]
        """
        # print(user_repr.mean(), cdd_news_repr.mean(), user_repr.max(), cdd_news_repr.max(), user_repr.sum(), cdd_news_repr.sum())
        score = cdd_news_repr.matmul(user_repr.transpose(-2,-1)).squeeze(-1)
        return score

    def _forward(self,x):
        his_news = x["his_encoded_index"].long().to(self.device)
        his_news_embedding = self.embedding(his_news)
        his_news_encoded_embedding, his_news_repr = self.encoderN(
            his_news_embedding
        )

        if self.reducer.name == 'matching':
            user_repr = self.encoderU(his_news_repr)
            ps_terms, ps_term_mask, kid = self.reducer(his_news_encoded_embedding, his_news_embedding, user_repr, his_news_repr, x["his_attn_mask"].to(self.device), x["his_reduced_mask"].to(self.device).bool())
        else:
            kid = None
            user_repr = None
            ps_terms, ps_term_mask = self.reducer(his_news_encoded_embedding, his_news_embedding, user_repr, his_news_repr, x["his_attn_mask"].to(self.device))

        # append CLS to each historical news, aggregator historical news representation to user repr
        if self.aggregator:
            ps_terms = torch.cat([his_news_embedding[:, :, 0].unsqueeze(-2), ps_terms], dim=-2)
            ps_term_mask = torch.cat([torch.ones(*ps_term_mask.shape[0:2], 1, device=ps_term_mask.device), ps_term_mask], dim=-1)
            ps_terms, his_news_repr = self.bert(ps_terms, ps_term_mask)
            user_repr = self.aggregator(his_news_repr)

        # append CLS to the entire browsing history, directly deriving user repr
        else:
            batch_size = ps_terms.size(0)
            ps_terms = torch.cat([his_news_embedding[:, 0, 0].unsqueeze(1).unsqueeze(1), ps_terms.reshape(batch_size, 1, -1, ps_terms.size(-1))], dim=-2)
            ps_term_mask = torch.cat([torch.ones(batch_size, 1, 1, device=ps_term_mask.device), ps_term_mask.reshape(batch_size, 1, -1)], dim=-1)
            _, user_cls = self.bert(ps_terms, ps_term_mask)
            user_repr = self.userProject(user_cls)

        cdd_news = x["cdd_encoded_index"].long().to(self.device)
        _, cdd_news_repr = self.bert(
            self.embedding(cdd_news), x['cdd_attn_mask'].to(self.device)
        )

        return self.clickPredictor(cdd_news_repr, user_repr), kid

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