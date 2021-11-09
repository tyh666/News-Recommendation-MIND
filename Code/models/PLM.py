# Two tower baseline
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from .TwoTowerBaseModel import TwoTowerBaseModel
from .Encoders.BERT import BERT_Encoder
from models.UniLM.configuration_tnlrv3 import TuringNLRv3Config
from models.UniLM.modeling import TuringNLRv3ForSequenceClassification, relative_position_bucket

class PLM(TwoTowerBaseModel):
    """
    Tow tower model with selection

    1. encode candidate news with bert
    2. encode ps terms with the same bert, using [CLS] embedding as user representation
    3. predict by scaled dot product
    """
    def __init__(self, manager, encoderU):
        super().__init__(manager)

        self.encoderU = encoderU

        if manager.debias:
            self.userBias = nn.Parameter(torch.randn(1,manager.bert_dim))
            nn.init.xavier_normal_(self.userBias)

        manager.name = '__'.join(["plm", manager.bert, manager.encoderU, manager.granularity])

        if manager.bert == 'unilm':
            config = TuringNLRv3Config.from_pretrained(manager.unilm_config_path)
            bert = TuringNLRv3ForSequenceClassification.from_pretrained(manager.unilm_path, config=config).bert

        elif manager.bert == 'deberta':
            # add a pooler
            bert = AutoModel.from_pretrained(
                manager.get_bert_for_load(),
                cache_dir=manager.path + 'bert_cache/'
            )
            self.pooler = nn.Sequential(
                nn.Linear(manager.bert_dim, manager.bert_dim),
                nn.GELU()
            )

        else:
            bert = AutoModel.from_pretrained(
                manager.get_bert_for_load(),
                cache_dir=manager.path + 'bert_cache/'
            )

        self.bert = bert


    def clickPredictor(self, cdd_news_repr, user_repr):
        """ calculate batch of click probabolity

        Args:
            cdd_news_repr: news-level representation, [batch_size, cdd_size, hidden_dim]
            user_repr: user representation, [batch_size, 1, hidden_dim]

        Returns:
            score of each candidate news, [batch_size, cdd_size]
        """
        score = cdd_news_repr.matmul(user_repr.transpose(-2,-1)).squeeze(-1)/math.sqrt(cdd_news_repr.size(-1))
        return score


    def _forward(self,x):
        # destroy encoding and embedding outside of the model

        batch_size = x['cdd_encoded_index'].size(0)

        if self.granularity != 'token':
            cdd_size = x['cdd_subword_index'].size(1)

            if self.training:
                cdd_dest = self.cdd_dest[:batch_size, :cdd_size]
                his_dest = self.his_dest[:batch_size]

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
                # * cdd_mask to filter out padded cdd news
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

        else:
            cdd_subword_prefix = None
            his_subword_prefix = None
            cdd_attn_mask = x['cdd_attn_mask'].to(self.device)
            his_attn_mask = x["his_attn_mask"].to(self.device)

        cdd_news = x["cdd_encoded_index"].to(self.device).view(-1, self.signal_length)
        cdd_attn_mask = cdd_attn_mask.view(-1, self.signal_length)
        cdd_news_repr = self.bert(cdd_news, cdd_attn_mask)[-1]
        if hasattr(self, 'pooler'):
            cdd_news_repr = self.pooler(cdd_news_repr[:, 0])
        cdd_news_repr = cdd_news_repr.view(batch_size, -1, self.hidden_dim)

        his_news = x["his_encoded_index"].to(self.device).view(-1, self.signal_length)
        his_attn_mask = his_attn_mask.view(-1, self.signal_length)
        his_news_repr = self.bert(his_news, his_attn_mask)[-1]
        if hasattr(self, 'pooler'):
            his_news_repr = self.pooler(his_news_repr[:, 0])
        his_news_repr = his_news_repr.view(batch_size, -1, self.hidden_dim)

        user_repr = self.encoderU(his_news_repr)

        if hasattr(self, 'userBias'):
            user_repr = user_repr + self.userBias

        return self.clickPredictor(cdd_news_repr, user_repr),


    def forward(self,x):
        """
        Decoupled function, score is unormalized click score
        """
        score, = self._forward(x)

        if self.training:
            prob = nn.functional.log_softmax(score, dim=1)
        else:
            prob = torch.sigmoid(score)

        return prob, None


    def encode_news(self, x):
        """
        encode news of loader_news
        """
        # encode news with MIND_news
        batch_size = x['cdd_encoded_index'].size(0)
        if self.granularity != 'token':
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

        cdd_news = x["cdd_encoded_index"].to(self.device).view(-1, self.signal_length)
        cdd_attn_mask = cdd_attn_mask.view(-1, self.signal_length)
        cdd_news_repr = self.bert(cdd_news, cdd_attn_mask)[-1]
        if hasattr(self, 'pooler'):
            cdd_news_repr = self.pooler(cdd_news_repr[:, 0])
        cdd_news_repr = cdd_news_repr.view(batch_size, self.hidden_dim)
        return cdd_news_repr


    def predict_fast(self, x):
        """
        1. encode user
        2. look up candidate representation in the embedding matrix
        3. compute click probability
        """
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

        else:
            his_subword_prefix = None
            his_attn_mask = x["his_attn_mask"].to(self.device)

        # [bs, cs, hd]
        cdd_news_repr = self.news_reprs(x['cdd_id'].to(self.device))

        his_news_repr = self.news_reprs(x['his_id'].to(self.device))
        user_repr = self.encoderU(his_news_repr)

        if hasattr(self, 'userBias'):
            user_repr = user_repr + self.userBias

        scores = cdd_news_repr.matmul(user_repr.transpose(-1, -2)).squeeze(-1)/math.sqrt(cdd_news_repr.size(-1))
        logits = torch.sigmoid(scores)
        return logits


    def encode_user(self, x):
        batch_size = x['his_encoded_index'].size(0)

        if self.granularity != 'token':
            cdd_size = x['cdd_subword_index'].size(1)

            if self.training:
                cdd_dest = self.cdd_dest[:batch_size, :cdd_size]
                his_dest = self.his_dest[:batch_size]

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
                # * cdd_mask to filter out padded cdd news
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

        else:
            his_subword_prefix = None
            his_attn_mask = x["his_attn_mask"].to(self.device)

        his_news = x["his_encoded_index"].to(self.device).view(-1, self.signal_length)
        his_attn_mask = his_attn_mask.view(-1, self.signal_length)
        his_news_repr = self.bert(his_news, his_attn_mask)[-1]
        if hasattr(self, 'pooler'):
            his_news_repr = self.pooler(his_news_repr[:, 0])
        his_news_repr = his_news_repr.view(batch_size, -1, self.hidden_dim)

        user_repr = self.encoderU(his_news_repr)

        if hasattr(self, 'userBias'):
            user_repr = user_repr + self.userBias

        return user_repr


class PLM2(TwoTowerBaseModel):
    """
    Tow tower model with selection

    1. encode candidate news with bert
    2. encode ps terms with the same bert, using [CLS] embedding as user representation
    3. predict by scaled dot product
    """
    def __init__(self, manager, embedding, encoderU):
        super().__init__(manager)

        self.embedding = embedding
        self.encoderU = encoderU
        self.bert = BERT_Encoder(manager)

        if manager.debias:
            self.userBias = nn.Parameter(torch.randn(1,manager.bert_dim))
            nn.init.xavier_normal_(self.userBias)

        self.hidden_dim = manager.bert_dim

        self.granularity = manager.granularity
        if self.granularity != 'token':
            self.register_buffer('cdd_dest', torch.zeros((self.batch_size, self.impr_size, self.signal_length * self.signal_length)), persistent=False)
            if manager.reducer in ["bm25", "entity", "first"]:
                self.register_buffer('his_dest', torch.zeros((self.batch_size, self.his_size, (manager.k + 2) * (manager.k + 2))), persistent=False)
            else:
                self.register_buffer('his_dest', torch.zeros((self.batch_size, self.his_size, self.signal_length * self.signal_length)), persistent=False)


        manager.name = '__'.join(["plm2", manager.bert, manager.encoderU, manager.granularity])
        self.name = manager.name

    def clickPredictor(self, cdd_news_repr, user_repr):
        """ calculate batch of click probabolity

        Args:
            cdd_news_repr: news-level representation, [batch_size, cdd_size, hidden_dim]
            user_repr: user representation, [batch_size, 1, hidden_dim]

        Returns:
            score of each candidate news, [batch_size, cdd_size]
        """
        score = cdd_news_repr.matmul(user_repr.transpose(-2,-1)).squeeze(-1)/math.sqrt(cdd_news_repr.size(-1))
        return score


    def _forward(self,x):
        # destroy encoding and embedding outside of the model

        batch_size = x['cdd_encoded_index'].size(0)

        if self.granularity != 'token':
            cdd_size = x['cdd_subword_index'].size(1)

            if self.training:
                cdd_dest = self.cdd_dest[:batch_size, :cdd_size]
                his_dest = self.his_dest[:batch_size]

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
                # * cdd_mask to filter out padded cdd news
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

        else:
            cdd_subword_prefix = None
            his_subword_prefix = None
            cdd_attn_mask = x['cdd_attn_mask'].to(self.device)
            his_attn_mask = x["his_attn_mask"].to(self.device)

        cdd_news = x["cdd_encoded_index"].to(self.device)
        _, cdd_news_repr = self.bert(
            self.embedding(cdd_news, cdd_subword_prefix), cdd_attn_mask
        )

        his_news = x["his_encoded_index"].to(self.device)
        _, his_news_repr = self.bert(
            self.embedding(his_news, his_subword_prefix), his_attn_mask
        )

        user_repr = self.encoderU(his_news_repr)

        if hasattr(self, 'userBias'):
            user_repr = user_repr + self.userBias

        return self.clickPredictor(cdd_news_repr, user_repr),


    def forward(self,x):
        """
        Decoupled function, score is unormalized click score
        """
        score, = self._forward(x)

        if self.training:
            prob = nn.functional.log_softmax(score, dim=1)
        else:
            prob = torch.sigmoid(score)

        return prob, None


    def encode_news(self, x):
        """
        encode news of loader_news
        """
        # encode news with MIND_news
        batch_size = x['cdd_encoded_index'].size(0)
        if self.granularity != 'token':
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

        return cdd_news_repr.squeeze(1)


    def predict_fast(self, x):
        """
        1. encode user
        2. look up candidate representation in the embedding matrix
        3. compute click probability
        """
        batch_size = x['his_encoded_index'].size(0)
        if self.granularity != 'token':
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

        else:
            his_subword_prefix = None
            his_attn_mask = x["his_attn_mask"].to(self.device)

        # [bs, cs, hd]
        cdd_news_repr = self.news_reprs(x['cdd_id'].to(self.device))

        his_news_repr = self.news_reprs(x['his_id'].to(self.device))
        user_repr = self.encoderU(his_news_repr)

        if hasattr(self, 'userBias'):
            user_repr = user_repr + self.userBias

        scores = cdd_news_repr.matmul(user_repr.transpose(-1, -2)).squeeze(-1)/math.sqrt(cdd_news_repr.size(-1))
        logits = torch.sigmoid(scores)
        return logits


class PLM3(TwoTowerBaseModel):
    """
    Tow tower model with selection

    1. encode candidate news with bert
    2. encode ps terms with the same bert, using [CLS] embedding as user representation
    3. predict by scaled dot product
    """
    def __init__(self, manager, encoderU):
        from transformers import AutoModel
        super().__init__(manager)

        if manager.bert == 'unilm':
            config = TuringNLRv3Config.from_pretrained(manager.unilm_config_path)
            # config.pooler = None
            bert = TuringNLRv3ForSequenceClassification.from_pretrained(manager.unilm_path, config=config).bert

            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            # unique in UniLM
            self.rel_pos_bias = bert.rel_pos_bias

        else:
            bert = AutoModel.from_pretrained(
                manager.get_bert_for_load(),
                cache_dir=manager.path + 'bert_cache/'
            )

        # bert = AutoModel.from_pretrained(manager.get_bert_for_load(), cache_dir=manager.path + 'bert_cache/')
        self.bert_embeddings = bert.embeddings
        self.bert = bert.encoder
        self.bert_pooler = bert.pooler
        self.encoderU = encoderU

        if manager.debias:
            self.userBias = nn.Parameter(torch.randn(1,manager.bert_dim))
            nn.init.xavier_normal_(self.userBias)

        self.hidden_dim = manager.bert_dim

        self.granularity = manager.granularity
        if self.granularity != 'token':
            self.register_buffer('cdd_dest', torch.zeros((self.batch_size, self.impr_size, self.signal_length * self.signal_length)), persistent=False)
            if manager.reducer in ["bm25", "entity", "first"]:
                self.register_buffer('his_dest', torch.zeros((self.batch_size, self.his_size, (manager.k + 2) * (manager.k + 2))), persistent=False)
            else:
                self.register_buffer('his_dest', torch.zeros((self.batch_size, self.his_size, self.signal_length * self.signal_length)), persistent=False)


        manager.name = '__'.join(["plm3", manager.bert, manager.encoderU, manager.granularity])
        self.name = manager.name

    def clickPredictor(self, cdd_news_repr, user_repr):
        """ calculate batch of click probabolity

        Args:
            cdd_news_repr: news-level representation, [batch_size, cdd_size, hidden_dim]
            user_repr: user representation, [batch_size, 1, hidden_dim]

        Returns:
            score of each candidate news, [batch_size, cdd_size]
        """
        score = cdd_news_repr.matmul(user_repr.transpose(-2,-1)).squeeze(-1)/math.sqrt(cdd_news_repr.size(-1))
        return score


    def _forward(self,x):
        # destroy encoding and embedding outside of the model

        batch_size = x['cdd_encoded_index'].size(0)

        if self.granularity != 'token':
            cdd_size = x['cdd_subword_index'].size(1)

            if self.training:
                cdd_dest = self.cdd_dest[:batch_size, :cdd_size]
                his_dest = self.his_dest[:batch_size]

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
                # * cdd_mask to filter out padded cdd news
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

        else:
            cdd_subword_prefix = None
            his_subword_prefix = None
            cdd_attn_mask = ((1 - x['cdd_attn_mask'].to(self.device)) * -10000.).view(-1, 1, 1, self.signal_length)
            his_attn_mask = ((1 - x["his_attn_mask"].to(self.device)) * -10000.).view(-1, 1, 1, self.signal_length)

        cdd_news = x["cdd_encoded_index"].to(self.device).view(-1, self.signal_length)
        cdd_news_embedding = self.bert_embeddings(cdd_news)
        if type(cdd_news_embedding) is tuple:
            position_ids = cdd_news_embedding[1]
            cdd_news_embedding = cdd_news_embedding[0]
        if hasattr(self, 'rel_pos_bias'):
            rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
            rel_pos = relative_position_bucket(rel_pos_mat, num_buckets=self.rel_pos_bins, max_distance=self.max_rel_pos)
            rel_pos = F.one_hot(rel_pos, num_classes=self.rel_pos_bins).float()
            rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
            bert_output = self.bert(cdd_news_embedding, attention_mask=cdd_attn_mask, rel_pos=rel_pos)[0]
        else:
            # [bs, sl/term_num+2, hd]
            bert_output = self.bert(cdd_news_embedding, attention_mask=cdd_attn_mask)[0]
        cdd_news_repr = self.bert_pooler(bert_output).view(batch_size, -1, self.hidden_dim)

        his_news = x["his_encoded_index"].to(self.device).view(-1, self.signal_length)
        his_news_embedding = self.bert_embeddings(his_news)
        if type(his_news_embedding) is tuple:
            position_ids = his_news_embedding[1]
            his_news_embedding = his_news_embedding[0]

        if hasattr(self, 'rel_pos_bias'):
            rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
            rel_pos = relative_position_bucket(rel_pos_mat, num_buckets=self.rel_pos_bins, max_distance=self.max_rel_pos)
            rel_pos = F.one_hot(rel_pos, num_classes=self.rel_pos_bins).float()
            rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
            bert_output = self.bert(his_news_embedding, attention_mask=his_attn_mask, rel_pos=rel_pos)[0]
        else:
            # [bs, sl/term_num+2, hd]
            bert_output = self.bert(his_news_embedding, attention_mask=his_attn_mask)[0]
        his_news_repr = self.bert_pooler(bert_output).view(batch_size, -1, self.hidden_dim)

        user_repr = self.encoderU(his_news_repr)

        if hasattr(self, 'userBias'):
            user_repr = user_repr + self.userBias

        return self.clickPredictor(cdd_news_repr, user_repr),


    def forward(self,x):
        """
        Decoupled function, score is unormalized click score
        """
        score, = self._forward(x)

        if self.training:
            prob = nn.functional.log_softmax(score, dim=1)
        else:
            prob = torch.sigmoid(score)

        return prob, None