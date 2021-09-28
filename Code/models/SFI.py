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

        manager.name = '__'.join(['sfi', manager.embedding, manager.encoderN, manager.selector, manager.ranker])

        self.final_dim = ranker.hidden_dim + self.his_size

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
        cdd_news = x['cdd_encoded_index'].to(self.device)
        cdd_news_embedding = self.embedding(cdd_news)
        _, cdd_news_repr = self.encoder(
            cdd_news_embedding, x['cdd_attn_mask'].to(self.device)
        )

        his_news = x["his_encoded_index"].to(self.device)
        his_news_embedding = self.embedding(his_news)

        his_attn_mask = x['his_attn_mask'].to(self.device)
        _, his_news_repr = self.encoder(
            his_news_embedding, his_attn_mask
        )

        output = self.selector(cdd_news_repr, his_news_repr, his_news_embedding, his_attn_mask, x["his_mask"].to(self.device))

        itr_tensors = self.ranker(cdd_news_embedding, output[0], x["cdd_attn_mask"].to(self.device), output[1])
        repr_tensors = cdd_news_repr.unsqueeze(dim=2).matmul(his_news_repr.unsqueeze(dim=1).transpose(-2,-1)).squeeze(dim=-2)

        return self._click_predictor(itr_tensors, repr_tensors)

    def forward(self, x):
        score = self._forward(x)
        if self.training:
            logit = nn.functional.log_softmax(score, dim=1)
        else:
            logit = torch.sigmoid(score)
        return (logit,)