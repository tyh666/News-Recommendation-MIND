import torch
import torch.nn as nn
from transformers import AutoModel

class Bert_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = 'bert-encoder'

        self.level = config.level

        # dimension for the final output embedding/representation
        self.hidden_dim = 768

        self.bert = AutoModel.from_pretrained(
            config.bert,
            # output hidden embedding of each transformer layer
            output_hidden_states=True
        )

    def forward(self, news_batch, **kwargs):
        """ encode news with bert

        Args:
            news_batch: batch of news tokens, of size [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, level, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        output = self.bert(news_batch.view(-1, news_batch.shape[2]), attention_mask=kwargs['attn_mask'].view(-1, news_batch.shape[2]))
        # stack the last level dimensions to form multi-level embedding
        news_embedding = torch.stack(output['hidden_states'][-self.level:],dim=-2)
        # use the hidden state of the last layer of [CLS] as news representation
        news_repr = news_embedding[:,0,-1,:].view(news_batch.shape[0], news_batch.shape[1], self.hidden_dim)

        return news_embedding.view(news_batch.shape + (self.level, self.hidden_dim)), news_repr