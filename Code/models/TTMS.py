# Two tower baseline
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Encoders.BERT import BERT_Encoder

class TTMS(nn.Module):
    def __init__(self, config, embedding, encoderN, encoderU, reducer, aggregator=None):
        super().__init__()

        self.scale = config.scale
        self.cdd_size = config.cdd_size
        self.batch_size = config.batch_size
        self.his_size = config.his_size
        self.signal_length = config.signal_length
        self.device = config.device

        self.embedding = embedding
        self.encoderN = encoderN
        self.encoderU = encoderU

        self.reducer = reducer
        self.bert = BERT_Encoder(config)

        self.aggregator = aggregator

        self.register_buffer('cdd_dest', torch.zeros((self.batch_size, config.impr_size, self.signal_length * self.signal_length)), persistent=False)
        self.register_buffer('his_dest', torch.zeros((self.batch_size, self.his_size, self.signal_length * self.signal_length)), persistent=False)

        if not aggregator:
            self.userProject = nn.Sequential(
                nn.Linear(self.bert.hidden_dim, self.bert.hidden_dim),
                nn.Tanh()
            )

        self.name = '__'.join(['ttms', self.encoderN.name, self.encoderU.name, config.reducer])
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
        if self.reducer.name == 'matching':
            his_news = x["his_encoded_index"].long().to(self.device)
            his_news_embedding = self.embedding(his_news, his_subword_prefix)
            his_news_encoded_embedding, his_news_repr = self.encoderN(
                his_news_embedding
            )
            user_repr = self.encoderU(his_news_repr)

            his_attn_mask = his_subword_prefix.matmul(x["his_attn_mask"].to(self.device).float().unsqueeze(-1)).squeeze(-1)
            his_reduced_mask = his_subword_prefix.matmul(x["his_reduced_mask"].to(self.device).float().unsqueeze(-1)).squeeze(-1)
            ps_terms, ps_term_mask, kid = self.reducer(his_news_encoded_embedding, his_news_embedding, user_repr, his_news_repr, his_attn_mask, his_reduced_mask)

        elif self.reducer.name == 'bow':
            his_reduced_news = x["his_reduced_index"].long().to(self.device)
            his_news_embedding = self.embedding(his_reduced_news, bow=True)
            his_reduced_encoded_embedding, his_reduced_repr = self.encoderN(his_news_embedding)
            user_repr = self.encoderU(his_reduced_repr)
            ps_terms, ps_term_mask, kid = self.reducer(his_reduced_encoded_embedding, his_news_embedding, user_repr, his_reduced_repr, x["his_attn_mask"].to(self.device))
            del user_repr, his_reduced_encoded_embedding, his_reduced_repr

        elif self.reducer.name == 'bm25':
            his_news = x["his_reduced_index"].long().to(self.device)
            his_news_embedding = self.embedding(his_news)
            his_news_encoded_embedding, his_news_repr = self.encoderN(
                his_news_embedding
            )

            kid = None
            user_repr = None
            ps_terms, ps_term_mask = self.reducer(his_news_encoded_embedding, his_news_embedding, user_repr, his_news_repr, x["his_reduced_mask"].to(self.device))

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
            self.embedding(cdd_news, cdd_subword_prefix), cdd_subword_prefix.matmul(x['cdd_attn_mask'].to(self.device).float().unsqueeze(-1)).squeeze(-1)
            # x['cdd_attn_mask'].to(self.device)
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