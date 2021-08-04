import torch
import torch.nn as nn
from transformers import BertModel

class BERT_Embedding(nn.Module):
    """
        pretrained bert word embedding and position embedding
    """
    def __init__(self, config):
        super().__init__()
        self.name = 'bert'

        # dimension for the final output embedding/representation
        self.embedding_dim = 768
        config.embedding_dim = self.embedding_dim

        self.signal_length = 512
        config.signal_length = self.signal_length

        bert = BertModel.from_pretrained(
            config.bert
        )
        self.embedding = bert.embeddings.word_embeddings
        self.pos_embedding = bert.embeddings.position_embeddings.weight
        self.layerNorm = bert.embeddings.LayerNorm
        self.dropOut = bert.embeddings.dropout


    def forward(self, news_batch):
        """ encode news with bert

        Args:
            news_batch: batch of news tokens, of size [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, emedding_dim]
        """


        # [1,sl,ed]
        pos_embeds = self.pos_embedding.unsqueeze(0)
        # [bs, cs/hs, sl]
        word_embeds = self.embedding(news_batch)

        embedding = self.dropOut(self.layerNorm(pos_embeds + word_embeds))

        return embedding