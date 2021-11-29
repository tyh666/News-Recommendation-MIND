import os
import logging
import torch
import torch.nn as nn

class Pipeline_Encoder(nn.Module):
    def __init__(self, manager):
        super().__init__()
        self.name = 'pipeline-encoder'

        news_repr_path = 'data/tensors/news_repr_{}_{}-[{}].tensor'.format(manager.scale,manager.mode,manager.pipeline)

        if os.path.exists(news_repr_path):
            self.news_repr = nn.Embedding.from_pretrained(torch.load(news_repr_path), freeze=True)
        else:
            logger = logging.getLogger(__name__)
            logger.warning("No encoded news at '{}', please encode news first!".format())
            raise ValueError

        self.level = news_embedding.shape[-2]
        self.hidden_dim = news_embedding.shape[-1]
        self.DropOut = nn.Dropout(manager.dropout_p)

    def forward(self,news_batch,**kwargs):
        """ encode news by lookup table

        Args:
            news_batch: tensor of [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        news_repr = self.DropOut(self.news_repr(kwargs['news_id']))
        news_embedding = self.DropOut(self.news_embedding(kwargs['news_id']).view(news_batch.shape + (self.level, self.hidden_dim)))
        return news_embedding, news_repr