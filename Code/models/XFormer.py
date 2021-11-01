import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from .BaseModel import BaseModel

class XFormer(BaseModel):
    """
    Tow tower model with selection

    1. encode candidate news with bert
    2. encode ps terms with the same bert, using [CLS] embedding as user representation
    3. predict by scaled dot product
    """
    def __init__(self, manager):
        super().__init__(manager)

        self.bert = AutoModel.from_pretrained(
            manager.get_bert_for_load(),
            cache_dir=manager.path + 'bert_cache/'
        )

        self.bert_name = manager.bert
        self.max_length = manager.get_max_length_for_truncating()

        if manager.debias:
            self.userBias = nn.Parameter(torch.randn(1,manager.bert_dim))
            nn.init.xavier_normal_(self.userBias)

        self.hidden_dim = manager.bert_dim
        self.register_buffer('cls_token_id', torch.tensor([manager.get_special_token_id('[CLS]')]).view(1, 1), persistent=False)
        self.register_buffer('cls_token_mask', torch.tensor([1]).view(1, 1), persistent=False)

        self.granularity = manager.granularity
        if self.granularity != 'token':
            self.register_buffer('cdd_dest', torch.zeros((self.batch_size, self.impr_size, self.signal_length * self.signal_length)), persistent=False)
            if manager.reducer in ["bm25", "entity", "first"]:
                self.register_buffer('his_dest', torch.zeros((self.batch_size, self.his_size, (manager.k + 2) * (manager.k + 2))), persistent=False)
            else:
                self.register_buffer('his_dest', torch.zeros((self.batch_size, self.his_size, self.signal_length * self.signal_length)), persistent=False)


        manager.name = '__'.join(['xformer', manager.bert, manager.granularity, manager.k])
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

        cdd_news = x["cdd_encoded_index"].to(self.device).view(-1, self.signal_length)
        cdd_attn_mask = cdd_attn_mask.view(-1, self.signal_length)
        cdd_news_repr = self.bert(cdd_news, cdd_attn_mask).pooler_output
        cdd_news_repr = cdd_news_repr.view(batch_size, -1, self.hidden_dim)

        # do not add extra [SEP]
        his_news = x["his_encoded_index"].to(self.device)[:, :, 1:].reshape(batch_size, -1)[:, :self.max_length]
        his_news = torch.cat([self.cls_token_id.expand(batch_size, 1), his_news], dim=-1)

        his_attn_mask = his_attn_mask[:, :, 1:].reshape(batch_size, -1)[:, :self.max_length]
        his_attn_mask = torch.cat([self.cls_token_mask.expand(batch_size, 1), his_attn_mask], dim=-1)

        user_repr = self.bert(his_news, his_attn_mask).pooler_output
        user_repr = user_repr.unsqueeze(1)

        if hasattr(self, 'userBias'):
            user_repr = user_repr + self.userBias

        return self.clickPredictor(cdd_news_repr, user_repr), None


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


    def encode_news(self, x):
        """
        encode news of loader_news
        """
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

        cdd_news = x["cdd_encoded_index"].to(self.device).view(-1, self.signal_length)
        cdd_attn_mask = cdd_attn_mask.view(-1, self.signal_length)
        cdd_news_repr = self.bert(cdd_news, cdd_attn_mask).pooler_output

        return cdd_news_repr


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
            his_refined_mask = None
            if 'his_refined_mask' in x:
                his_refined_mask = his_subword_prefix.matmul(x["his_refined_mask"].to(self.device).float().unsqueeze(-1)).squeeze(-1)

        else:
            his_subword_prefix = None
            his_attn_mask = x["his_attn_mask"].to(self.device)
            his_refined_mask = None
            if 'his_refined_mask' in x:
                his_refined_mask = x["his_refined_mask"].to(self.device)

        his_news = x["his_encoded_index"].to(self.device)[:, :, 1:].reshape(batch_size, -1)[:, :self.max_length]
        his_news = torch.cat([self.cls_token_id.expand(batch_size, 1), his_news], dim=-1)

        his_attn_mask = his_attn_mask[:, :, 1:].reshape(batch_size, -1)[:, :self.max_length]
        his_attn_mask = torch.cat([self.cls_token_mask.expand(batch_size, 1), his_attn_mask], dim=-1)

        user_repr = self.bert(his_news, his_attn_mask).pooler_output
        user_repr = user_repr.unsqueeze(1)

        if hasattr(self, 'userBias'):
            user_repr = user_repr + self.userBias

        # [bs, cs, hd]
        news_repr = self.news_reprs(x['cdd_id'].to(self.device))

        scores = news_repr.matmul(user_repr.transpose(-1, -2)).squeeze(-1)/math.sqrt(news_repr.size(-1))
        logits = torch.sigmoid(scores)
        return logits