# Two tower baseline
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Encoders.BERT import BERT_Encoder


class EncoderContainer(nn.Module):
    """
    wrap news and user encoder
    """
    def __init__(self, manager, embedding, encoderN, encoderU, reducer):
        super().__init__()

        self.scale = manager.scale
        self.his_size = manager.his_size
        self.signal_length = manager.signal_length
        self.device = manager.device

        self.embedding = embedding
        self.encoderN = encoderN
        self.encoderU = encoderU

        self.reducer = reducer
        self.bert = BERT_Encoder(manager)
        self.newsUserProject = nn.Sequential(
            nn.Linear(self.bert.hidden_dim, self.bert.hidden_dim),
            nn.Tanh()
        )
        if manager.debias:
            self.userBias = nn.Parameter(torch.randn(1,self.bert.hidden_dim))
            nn.init.xavier_normal_(self.userBias)

        self.hidden_dim = self.bert.hidden_dim

        self.granularity = manager.granularity
        if self.granularity != 'token':
            self.register_buffer('cdd_dest', torch.zeros((manager.batch_size_news, manager.signal_length * manager.signal_length)), persistent=False)
            if manager.reducer in ["bm25", "entity", "first"]:
                self.register_buffer('his_dest', torch.zeros((manager.batch_size_history, self.his_size, (manager.k + 1) * (manager.k + 1))), persistent=False)
            else:
                self.register_buffer('his_dest', torch.zeros((manager.batch_size_history, self.his_size, manager.signal_length * manager.signal_length)), persistent=False)

        manager.name = '__'.join(['ttms', manager.embedding, manager.encoderN, manager.encoderU, manager.reducer, manager.granularity])


    def forward(self,x):
        if 'his_encoded_index' in x:
            if self.granularity != 'token':
                batch_size = x['his_encoded_index'].size(0)
                his_dest = self.his_dest[:batch_size]

                his_subword_index = x['his_subword_index'].to(self.device)
                his_signal_length = his_subword_index.size(-2)
                his_subword_index = his_subword_index[:, :, :, 0] * his_signal_length + his_subword_index[:, :, :, 1]

                his_subword_prefix = his_dest.scatter(dim=-1, index=his_subword_index, value=1) * x["his_mask"].to(self.device)
                his_subword_prefix = his_subword_prefix.view(batch_size, self.his_size, his_signal_length, his_signal_length)

                if self.granularity == 'avg':
                    # average subword embeddings as the word embedding
                    his_subword_prefix = F.normalize(his_subword_prefix, p=1, dim=-1)

                his_attn_mask = his_subword_prefix.matmul(x["his_attn_mask"].to(self.device).float().unsqueeze(-1)).squeeze(-1)
                his_refined_mask = None
                if 'his_refined_mask' in x:
                    his_refined_mask = his_subword_prefix.matmul(x["his_refined_mask"].to(self.device).float().unsqueeze(-1)).squeeze(-1)

            else:
                his_subword_prefix = None
                his_attn_mask = x["his_attn_mask"].to(self.device)
                his_refined_mask = None
                if 'his_refined_mask' in x:
                    his_refined_mask = x["his_refined_mask"].to(self.device)

            his_news = x["his_encoded_index"].to(self.device)
            his_news_embedding = self.embedding(his_news, his_subword_prefix)
            his_news_encoded_embedding, his_news_repr = self.encoderN(
                his_news_embedding, his_attn_mask
            )
            # no need to calculate this if ps_terms are fixed in advance
            if self.reducer.name == 'matching':
                user_repr = self.encoderU(his_news_repr, his_mask=x['his_mask'].to(self.device), user_index=x['user_id'].to(self.device))
            else:
                user_repr = None

            ps_terms, ps_term_mask, _ = self.reducer(his_news_encoded_embedding, his_news_embedding, user_repr, his_news_repr, his_attn_mask, his_refined_mask)

            _, user_cls = self.bert(ps_terms, ps_term_mask)
            user_repr = self.newsUserProject(user_cls.squeeze(1))
            if hasattr(self, 'userBias'):
                user_repr = user_repr + self.userBias
            return user_repr

        else:
            # encode news with MIND_news
            if self.granularity != 'token':
                batch_size = x['cdd_subword_index'].size(0)
                cdd_dest = self.cdd_dest[:batch_size]
                cdd_subword_index = x['cdd_subword_index'].to(self.device)
                cdd_subword_index = cdd_subword_index[:, :, 0] * self.signal_length + cdd_subword_index[:, :, 1]
                cdd_subword_prefix = cdd_dest.scatter(dim=-1, index=cdd_subword_index, value=1)

                cdd_subword_prefix = cdd_subword_prefix.view(batch_size, self.signal_length, self.signal_length)

                if self.granularity == 'avg':
                    # average subword embeddings as the word embedding
                    cdd_subword_prefix = F.normalize(cdd_subword_prefix, p=1, dim=-1)

                cdd_attn_mask = cdd_subword_prefix.matmul(x['cdd_attn_mask'].to(self.device).float().unsqueeze(-1)).squeeze(-1)

            else:
                cdd_subword_prefix = None
                cdd_attn_mask = x['cdd_attn_mask'].to(self.device)

            cdd_news = x["cdd_encoded_index"].to(self.device)
            _, cdd_news_repr = self.bert(
                self.embedding(cdd_news, cdd_subword_prefix), cdd_attn_mask
            )
            cdd_news_repr = self.newsUserProject(cdd_news_repr.squeeze(1))

            return cdd_news_repr


class TestContainer(nn.Module):
    """
    wrap fast test
    """
    def __init__(self, manager):
        super().__init__()
        self.device = manager.device
        self.cache_directory = "data/cache/{}/{}/".format(manager.name, manager.scale)

    def _init_embedding(self):
        self.news_reprs = nn.Embedding.from_pretrained(torch.load(self.cache_directory + "news.pt", map_location=torch.device(self.device)))
        self.user_reprs = nn.Embedding.from_pretrained(torch.load(self.cache_directory + "user.pt", map_location=torch.device(self.device)))

    def forward(self, x):
        if not hasattr(self, 'news_reprs'):
            self._init_embedding()

        # bs, 1, hd
        user_repr = self.user_reprs(x['user_id'].to(self.device)).unsqueeze(-1)
        # bs, cs, hd
        news_repr = self.news_reprs(x['cdd_id'].to(self.device))

        scores = news_repr.matmul(user_repr).squeeze(-1)
        logits = torch.sigmoid(scores)
        return logits