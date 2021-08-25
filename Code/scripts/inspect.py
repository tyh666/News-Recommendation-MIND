import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import prepare
from data.configs.demo import config

from utils.Manager import Manager

from models.Embeddings.BERT import BERT_Embedding
from models.Encoders.CNN import CNN_Encoder,CNN_User_Encoder
from models.Encoders.RNN import RNN_Encoder,RNN_User_Encoder
from models.Encoders.MHA import MHA_Encoder, MHA_User_Encoder
from models.Encoders.BERT import BERT_Encoder
from models.Modules.DRM import Matching_Reducer, BM25_Reducer
from models.Rankers.BERT import BERT_Onepass_Ranker, BERT_Original_Ranker
from models.Rankers.CNN import CNN_Ranker
from models.ESM import ESM
from models.TTMS import TTMS

manager = Manager(config)
loaders = prepare(manager)

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
        score_coarse = cdd_news_repr.matmul(user_repr.transpose(-2,-1))
        score = torch.cat([reduced_tensor, score_coarse], dim=-1)

        return self.learningToRank(score).squeeze(dim=-1)

    def _forward(self,x):
        cdd_news = x["cdd_encoded_index"].long().to(self.device)
        cdd_news_embedding = self.embedding(cdd_news)
        _, cdd_news_repr = self.encoderN(
            cdd_news_embedding
        )
        if self.reducer.name == 'bm25':
            his_news = x["his_reduced_index"].long().to(self.device)
        else:
            his_news = x["his_encoded_index"].long().to(self.device)
        his_news_embedding = self.embedding(his_news)
        his_news_encoded_embedding, his_news_repr = self.encoderN(
            his_news_embedding
        )

        user_repr = self.encoderU(his_news_repr)
        if self.reducer.name == 'matching':
            ps_terms, ps_term_mask, score_kid = self.reducer(his_news_encoded_embedding, his_news_embedding, user_repr, his_news_repr, x["his_attn_mask"].to(self.device), x["his_attn_mask_k"].to(self.device).bool())

        else:
            ps_terms, ps_term_mask = self.reducer(his_news_encoded_embedding, his_news_embedding, user_repr, x["his_attn_mask"].to(self.device))

        if self.fuser:
            ps_terms, ps_term_mask = self.fuser(ps_terms, ps_term_mask)

        # reduced_tensor = self.ranker(torch.cat([cdd_news_repr.unsqueeze(-2), cdd_news_embedding], dim=-2), torch.cat([user_repr, ps_terms], dim=-2))

        reduced_tensor = self.ranker(cdd_news_embedding, ps_terms, x["cdd_attn_mask"].to(self.device), ps_term_mask)

        return self.clickPredictor(reduced_tensor, cdd_news_repr, user_repr), score_kid

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

class TTMS(nn.Module):
    def __init__(self, config, embedding, encoderN, encoderU, aggregator=None):
        super().__init__()

        self.scale = config.scale
        self.cdd_size = config.cdd_size
        self.batch_size = config.batch_size
        self.his_size = config.his_size
        self.device = config.device

        self.embedding = embedding
        self.encoderN = encoderN
        self.encoderU = encoderU

        self.reducer = Matching_Reducer(config)
        self.bert = BERT_Encoder(config)

        config.hidden_dim = config.embedding_dim
        self.aggregator = aggregator
        if not aggregator:
            self.userProject = nn.Sequential(
                nn.Linear(self.bert.hidden_dim, self.bert.hidden_dim),
                nn.Tanh()
            )

        self.name = '__'.join(['ttms', self.encoderN.name, self.encoderU.name])
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

        user_repr = self.encoderU(his_news_repr)

        ps_terms, ps_term_mask, kid = self.reducer(his_news_encoded_embedding, his_news_embedding, user_repr, his_news_repr, x["his_attn_mask"].to(self.device), x["his_attn_mask_k"].to(self.device).bool())

        # append CLS to each historical news, aggregator historical news representation to user repr
        # ps_terms = torch.cat([his_news_embedding[:, :, 0].unsqueeze(-2), ps_terms], dim=-2)
        # ps_term_mask = torch.cat([torch.ones(*ps_term_mask.shape[0:2], 1, device=ps_term_mask.device), ps_term_mask], dim=-1)
        # ps_terms, his_news_repr = self.bert(ps_terms, ps_term_mask)
        # user_repr = self.aggregator(his_news_repr)

        # append CLS to the entire browsing history, directly deriving user repr
        batch_size = ps_terms.size(0)
        ps_terms = torch.cat([his_news_embedding[:, 0, 0].unsqueeze(1).unsqueeze(1), ps_terms.view(batch_size, 1, -1, ps_terms.size(-1))], dim=-2)
        ps_term_mask = torch.cat([torch.ones(batch_size, 1, 1, device=ps_term_mask.device), ps_term_mask.view(batch_size, 1, -1)], dim=-1)
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

class Matching_Reducer(nn.Module):
    """
    basic document reducer: topk of each historical article
    """
    def __init__(self, config):
        super().__init__()

        self.name = "matching"

        self.k = config.k
        self.diversify = config.diversify

        config.term_num = config.k * config.his_size

        if self.diversify:
            self.newsUserAlign = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
            nn.init.xavier_normal_(self.newsUserAlign.weight)

        if config.threshold != -float('inf'):
            threshold = torch.tensor([config.threshold])
            self.register_buffer('threshold', threshold)

    def forward(self, news_selection_embedding, news_embedding, user_repr, news_repr, his_attn_mask, his_attn_mask_k):
        """
        Extract words from news text according to the overall user interest

        Args:
            news_selection_embedding: encoded word-level embedding, [batch_size, his_size, signal_length, hidden_dim]
            news_embedding: word-level news embedding, [batch_size, his_size, signal_length, hidden_dim]
            news_repr: news-level representation, [batch_size, his_size, hidden_dim]
            user_repr: user representation, [batch_size, 1, hidden_dim]

        Returns:
            ps_terms: weighted embedding for personalized terms, [batch_size, his_size, k, hidden_dim]
            ps_term_mask: attention mask of output terms, [batch_size, his_size, k]
        """
        # strip off [CLS]
        news_selection_embedding = news_selection_embedding[:, :, 1:]
        news_embedding = news_embedding[:, :, 1:]
        if self.diversify:
            news_user_repr = torch.cat([user_repr.expand(news_repr.size()), news_repr], dim=-1)
            selection_query = self.newsUserAlign(news_user_repr).unsqueeze(-1)
        else:
            selection_query = user_repr.expand(news_repr.size()).unsqueeze(-1)

        # [bs, hs, sl - 1]
        scores = F.normalize(news_selection_embedding, dim=-1).matmul(F.normalize(selection_query, dim=-1)).squeeze(-1)
        # mask the padded term
        scores = scores.masked_fill(~his_attn_mask_k[:, :, 1:], -float('inf'))

        score_k, score_kid = scores.topk(dim=-1, k=self.k, sorted=False)

        ps_terms = news_embedding.gather(dim=-2,index=score_kid.unsqueeze(-1).expand(score_kid.size() + (news_embedding.size(-1),)))
        # [bs, hs, k]
        ps_term_mask = his_attn_mask[:, :, 1:].gather(dim=-1, index=score_kid)

        if hasattr(self, 'threshold'):
            mask_pos = score_k < self.threshold
            # ps_terms = personalized_terms * (nn.functional.softmax(score_k.masked_fill(score_k < self.threshold, 0), dim=-1).unsqueeze(-1))
            ps_terms = ps_terms * (score_k.masked_fill(mask_pos, 0).unsqueeze(-1))
            ps_term_mask = ps_term_mask * (~mask_pos)

        return ps_terms, ps_term_mask, score_kid

embedding = BERT_Embedding(manager)

encoderN = CNN_Encoder(manager)
# encoderN = RNN_Encoder(manager)
# encoderN = MHA_Encoder(manager)

# encoderU = CNN_User_Encoder(manager)
encoderU = RNN_User_Encoder(manager)
# encoderU = MHA_User_Encoder(manager)

docReducer = Matching_Reducer(manager)
# docReducer = BM25_Reducer(manager)

# ranker = CNN_Ranker(manager)
# ranker = BERT_Onepass_Ranker(manager)
# ranker = BERT_Original_Ranker(manager)

# model = ESM(manager, embedding, encoderN, encoderU, docReducer, None, ranker).to(manager.device)
model = TTMS(manager, embedding, encoderN, encoderU)

manager.inspect_ps_terms(model, 4123, loaders[0], bm25=True)