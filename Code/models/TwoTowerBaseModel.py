import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerBaseModel(nn.Module):
    def __init__(self, manager):
        super().__init__()

        self.scale = manager.scale
        self.cdd_size = manager.cdd_size
        self.mode = "test" if manager.mode == "test" else "dev"

        self.impr_size = manager.impr_size
        # used for encoding
        self.batch_size_news = manager.batch_size_news
        # encoding flag set to false
        self.encoding = False

        self.his_size = manager.his_size
        self.signal_length = manager.signal_length
        self.device = manager.device

        self.hidden_dim = manager.bert_dim

        self.granularity = manager.granularity
        if self.granularity != 'token':
            self.register_buffer('cdd_dest', torch.zeros((manager.batch_size, self.impr_size, self.signal_length * self.signal_length)), persistent=False)
            if manager.reducer in ["bm25", "entity", "first"]:
                self.register_buffer('his_dest', torch.zeros((manager.batch_size, self.his_size, (manager.k + 1) * (manager.k + 1))), persistent=False)
            else:
                self.register_buffer('his_dest', torch.zeros((manager.batch_size, self.his_size, self.signal_length * self.signal_length)), persistent=False)


    def init_encoding(self):
        """
        prepare for fast encoding
        """
        if self.granularity != 'token':
            self.cdd_dest = torch.zeros((self.batch_size_news, self.signal_length * self.signal_length), device=self.device)
        self.encoding = True


    def init_embedding(self):
        """
        prepare for fast inferring
        """
        self.cache_directory = "data/cache/tensors/{}/{}/{}/".format(self.name, self.scale, self.mode)
        self.news_reprs = nn.Embedding.from_pretrained(torch.load(self.cache_directory + "news.pt", map_location=torch.device(self.device)))


    def destroy_encoding(self):
        if self.granularity != 'token':
            self.cdd_dest = torch.zeros((self.batch_size, self.impr_size, self.signal_length * self.signal_length), device=self.device)
        self.encoding = False


    def destroy_embedding(self):
        self.news_reprs = None
        del self.news_reprs


    def compute_score(self, cdd_news_repr, user_repr):
        """ calculate batch of click probabolity

        Args:
            cdd_news_repr: news-level representation, [batch_size, cdd_size, hidden_dim]
            user_repr: user representation, [batch_size, 1, hidden_dim]

        Returns:
            score of each candidate news, [batch_size, cdd_size]
        """
        score = cdd_news_repr.matmul(user_repr.transpose(-2,-1)).squeeze(-1)/math.sqrt(cdd_news_repr.size(-1))
        return score


    def forward(self,x):
        cdd_repr = self.encode_news(x)
        user_repr, kid = self.encode_user(x)
        score = self.compute_score(cdd_repr, user_repr)
        print(score)

        return
        if self.training:
            logits = nn.functional.log_softmax(score, dim=1)
        else:
            logits = torch.sigmoid(score)

        return logits, kid


    def predict_fast(self, x):
        # [bs, cs, hd]
        cdd_repr = self.news_reprs(x['cdd_id'].to(self.device))
        user_repr, _ = self.encode_user(x)
        scores = self.compute_score(cdd_repr, user_repr)
        logits = torch.sigmoid(scores)
        return logits