# selective fine-grained interaction framework
import torch
import torch.nn as nn


class SFI(nn.Module):
    def __init__(self, manager, embedding, encoder, selector, ranker):
        super().__init__()

        self.scale = manager.scale
        self.batch_size = manager.batch_size
        self.his_size = manager.his_size
        self.signal_length = manager.signal_length
        self.device = manager.device

        self.k = manager.k

        self.embedding = embedding
        self.encoder = encoder
        self.selector = selector
        self.ranker = ranker

        self.name = '__'.join(['sfi', self.encoder.name, self.ranker.name])
        manager.name = self.name

        self.hidden_dim = encoder.hidden_dim
        self.final_dim = ranker.final_dim + self.his_size

        self.learningToRank = nn.Sequential(
            nn.Linear(self.final_dim, int(self.final_dim/2)),
            nn.ReLU(),
            nn.Linear(int(self.final_dim/2),1)
        )

        for param in self.learningToRank:
            if isinstance(param, nn.Linear):
                nn.init.xavier_normal_(param.weight)


    def _click_predictor(self, itr_tensors, repr_tensors):
        """ calculate batch of click probabolity

        Args:
            fusion_tensors: tensor of [batch_size, cdd_size, *]

        Returns:
            score: tensor of [batch_size, cdd_size], which is normalized click probabilty
        """
        score = self.learningToRank(torch.cat([itr_tensors, repr_tensors], dim=-1)).squeeze(dim=-1)
        return score

    def _forward(self, x):
        cdd_news = x['cdd_encoded_index'].long().to(self.device)
        cdd_news_embedding = self.embedding(cdd_news)
        _, cdd_news_repr = self.encoder(
            cdd_news_embedding
        )
            # user_index=x['user_index'].long().to(self.device),
            # news_id=x['cdd_id'].long().to(self.device))

        his_news = x["his_encoded_index"].long().to(self.device)
        his_news_embedding = self.embedding(his_news)
        _, his_news_repr = self.encoder(
            his_news_embedding,
        )
            # user_index=x['user_index'].long().to(self.device),
            # news_id=x['his_id'].long().to(self.device))
            # attn_mask=x['clicked_title_pad'].to(self.device))

        # t2 = time.time()
        output = self.selector(cdd_news_repr, his_news_repr, his_news_embedding, x["his_attn_mask"].to(self.device))
        # t3 = time.time()

        itr_tensors = self.ranker(cdd_news_embedding, output[0], x["cdd_attn_mask"].to(self.device), output[1])
        repr_tensors = cdd_news_repr.unsqueeze(dim=2).matmul(his_news_repr.unsqueeze(dim=1).transpose(-2,-1)).squeeze(dim=-2)
        # t4 = time.time()
        # print("encoding time:{} selection time:{} interacting time:{}".format(t2-t1, t3-t2, t4-t3))

        return self._click_predictor(itr_tensors, repr_tensors)

    def forward(self, x):
        score = self._forward(x)
        if self.training:
            score = nn.functional.log_softmax(score, dim=1)
        else:
            score = torch.sigmoid(score)
        return (score,)