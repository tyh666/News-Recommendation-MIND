import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class OneTowerGateFormer(nn.Module):
    def __init__(self, manager, embedding, encoderN, encoderU, reducer, ranker):
        super().__init__()
        self.his_size = manager.his_size
        self.signal_length = manager.signal_length
        self.device = manager.device
        self.hidden_dim = manager.bert_dim

        self.embedding = embedding
        # personalized keywords selection
        if manager.reducer == "personalized":
            self.encoderN = encoderN
            self.encoderU = encoderU
        # global keywords selection
        elif manager.reducer == "global":
            self.encoderN = encoderN
            self.query = nn.Parameter(torch.randn(1, manager.bert_dim))
            self.queryProject = nn.Linear(manager.bert_dim, encoderN.hidden_dim)
            nn.init.xavier_uniform_(self.query)
        self.reducer = reducer
        self.bert = ranker
        self.learning2Rank = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )

        self.reducer_name = manager.reducer

        # global selection
        # self.query = nn.Parameter(torch.randn(1, manager.bert_dim))
        # self.queryProject = nn.Linear(manager.bert_dim, encoderN.hidden_dim)
        # nn.init.xavier_uniform_(self.query)


        manager.name = '__'.join(['esm', manager.bert, manager.encoderN, manager.encoderU, manager.reducer, manager.ranker, str(manager.k), manager.verbose])
        # used in fast evaluate
        self.name = manager.name


    def encode_impression(self, x):
        """
        encode candidate news
        """
        batch_size = x['his_encoded_index'].size(0)
        # encode news with MIND_news
        cdd_news = x["cdd_encoded_index"].to(self.device)
        cdd_attn_mask = x['cdd_attn_mask'].to(self.device)
        cdd_news_embedding = self.embedding(cdd_news)

        his_news = x["his_encoded_index"].to(self.device)
        his_attn_mask = x["his_attn_mask"].to(self.device)
        if 'his_refined_mask' in x:
            his_refined_mask = x["his_refined_mask"].to(self.device)

        his_news_embedding = self.embedding(his_news)
        if hasattr(self, 'encoderN'):
            his_news_encoded_embedding,  his_news_repr = self.encoderN(
                his_news_embedding, his_attn_mask
            )
        else:
            his_news_encoded_embedding = None
            his_news_repr = None

        if self.reducer_name == 'personalized':
            user_repr_ext = self.encoderU(his_news_repr, his_mask=x['his_mask'], user_index=x['user_id'].to(self.device))
        elif self.reducer_name == "global":
            user_repr_ext = self.queryProject(self.query).expand(batch_size, 1, -1)
        else:
            user_repr_ext = None

        ps_terms, ps_term_mask, kid = self.reducer(his_news_encoded_embedding, his_news_embedding, user_repr_ext, his_news_repr, his_attn_mask, his_refined_mask)

        fusion_tensor = self.bert(cdd_news_embedding, ps_terms, cdd_attn_mask, ps_term_mask)
        return fusion_tensor, kid


    def compute_score(self, fusion_tensor):
        """
        compute click score
        """
        return self.learning2Rank(fusion_tensor).squeeze(-1)


    def forward(self, x):
        fusion_tensor, kid = self.encode_impression(x)
        score = self.compute_score(fusion_tensor)
        if self.training:
            logits = nn.functional.log_softmax(score, dim=1)
        else:
            logits = torch.sigmoid(score)

        return logits, kid