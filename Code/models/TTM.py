# Two tower baseline
import torch
import torch.nn as nn

class TTM(nn.Module):
    def __init__(self, manager, embedding, encoderN, encoderU):
        super().__init__()

        self.scale = manager.scale
        self.cdd_size = manager.cdd_size
        self.batch_size = manager.batch_size
        self.his_size = manager.his_size
        self.device = manager.device

        self.embedding = embedding
        self.encoderN = encoderN
        self.encoderU = encoderU

        self.reducer = manager.reducer

        manager.name = '__'.join(['ttm', manager.embedding, manager.encoderN, manager.encoderU, manager.granularity])


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
        cdd_news = x["cdd_encoded_index"].long().to(self.device)
        _, cdd_news_repr = self.encoderN(
            self.embedding(cdd_news), x['cdd_attn_mask'].to(self.device)
        )

        his_news = x["his_encoded_index"].long().to(self.device)
        _, his_news_repr = self.encoderN(
            self.embedding(his_news), x['his_attn_mask'].to(self.device)
        )

        user_repr = self.encoderU(his_news_repr)

        return self.clickPredictor(cdd_news_repr, user_repr)

    def forward(self,x):
        """
        Decoupled function, score is unormalized click score
        """
        score = self._forward(x)

        if self.training:
            prob = nn.functional.log_softmax(score, dim=1)
        else:
            prob = torch.sigmoid(score)

        return (prob,)