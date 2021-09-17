# Two tower baseline
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Encoders.BERT import BERT_Encoder

class TTMS(nn.Module):
    """
    Tow tower model with selection

    1. encode candidate news with bert
    2. encode ps terms with the same bert, using [CLS] embedding as user representation
    3. predict by scaled dot product
    """
    def __init__(self, config, embedding, encoderN, encoderU, reducer):
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

        self.granularity = config.granularity
        if self.granularity != 'token':
            self.register_buffer('cdd_dest', torch.zeros((self.batch_size, config.impr_size, config.signal_length * config.signal_length)), persistent=False)
            if config.reducer in ["bm25", "entity", "first"]:
                self.register_buffer('his_dest', torch.zeros((self.batch_size, self.his_size, (config.k + 1) * (config.k + 1))), persistent=False)
            else:
                self.register_buffer('his_dest', torch.zeros((self.batch_size, self.his_size, config.signal_length * config.signal_length)), persistent=False)

        self.newsUserProject = nn.Sequential(
            nn.Linear(self.bert.hidden_dim, self.bert.hidden_dim),
            nn.Tanh()
        )

        if config.debias:
            self.userBias = nn.Parameter(torch.randn(1,self.bert.hidden_dim))
            nn.init.xavier_normal_(self.userBias)

        self.register_buffer('extra_cls_mask', torch.ones(1,1), persistent=False)

        config.name = '__'.join(['ttms', config.embedding, config.encoderN, config.encoderU, config.reducer, config.granularity])


    def clickPredictor(self, cdd_news_repr, user_repr):
        """ calculate batch of click probabolity

        Args:
            cdd_news_repr: news-level representation, [batch_size, cdd_size, hidden_dim]
            user_repr: user representation, [batch_size, 1, hidden_dim]

        Returns:
            score of each candidate news, [batch_size, cdd_size]
        """
        if hasattr(self, 'newsDebias'):
            user_repr += self.userBias
        score = cdd_news_repr.matmul(user_repr.transpose(-2,-1)).squeeze(-1)/math.sqrt(self.embedding.embedding_dim)
        return score

    def _forward(self,x):
        if self.granularity != 'token':
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
            his_subword_index = x['his_subword_index'].to(self.device)
            his_signal_length = his_subword_index.size(-2)
            cdd_subword_index = cdd_subword_index[:, :, :, 0] * self.signal_length + cdd_subword_index[:, :, :, 1]
            his_subword_index = his_subword_index[:, :, :, 0] * his_signal_length + his_subword_index[:, :, :, 1]

            if self.training:
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
        _, cdd_news_repr = self.bert(
            self.embedding(cdd_news, cdd_subword_prefix), cdd_attn_mask
        )
        cdd_news_repr = self.newsUserProject(cdd_news_repr)

        his_news = x["his_encoded_index"].long().to(self.device)
        his_news_embedding = self.embedding(his_news, his_subword_prefix)
        his_news_encoded_embedding, his_news_repr = self.encoderN(
            his_news_embedding, his_attn_mask
        )
        # no need to calculate this if ps_terms are fixed in advance
        if self.reducer.name == 'matching':
            user_repr = self.encoderU(his_news_repr)
        else:
            user_repr = None

        ps_terms, ps_term_mask, kid = self.reducer(his_news_encoded_embedding, his_news_embedding, user_repr, his_news_repr, his_attn_mask, his_refined_mask)

        _, user_cls = self.bert(ps_terms, ps_term_mask)
        user_repr = self.newsUserProject(user_cls)

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