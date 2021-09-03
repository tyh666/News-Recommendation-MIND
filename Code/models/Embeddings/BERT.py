import torch.nn as nn
from transformers import AutoModel

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

        bert = AutoModel.from_pretrained(
            config.bert,
            cache_dir=config.path + 'bert_cache/'
        )
        self.embedding = bert.embeddings.word_embeddings
        # [1, (1,) *, embedding_dim]
        try:
            self.pos_embedding = nn.Parameter(bert.embeddings.position_embeddings.weight[:config.signal_length])
        except:
            self.pos_embedding = None
        # self.token_type_embedding = nn.Parameter(bert.embeddings.token_type_embeddings.weight[0])

        self.layerNorm = bert.embeddings.LayerNorm
        self.dropOut = bert.embeddings.dropout

        if config.reducer == 'bow':
            self.freq_embedding = nn.Embedding(config.signal_length // 2, self.embedding_dim)
            nn.init.xavier_normal_(self.freq_embedding.weight)

    def forward(self, news_batch, subword_prefix=None):
        """ encode news with bert

        Args:
            news_batch: batch of news tokens, of size [batch_size, *, signal_length]
            bow: whether the input is bow

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, emedding_dim]
        """
        if news_batch.dim() == 4:
            if self.pos_embedding is not None:
                word_embeds = self.embedding(news_batch[:,:,:,0]) + self.freq_embedding(news_batch[:,:,:,1]) + self.pos_embedding[:news_batch.size(2)]
            else:
                word_embeds = self.embedding(news_batch[:,:,:,0]) + self.freq_embedding(news_batch[:,:,:,1])

        else:
            # [bs, cs/hs, sl, ed]
            if self.pos_embedding is not None:
                word_embeds = self.embedding(news_batch) + self.pos_embedding[:news_batch.size(2)]
            else:
                word_embeds = self.embedding(news_batch)

        # word-level
        if subword_prefix is not None:
            word_embeds = subword_prefix.matmul(word_embeds)

        # embedding = self.dropOut(self.layerNorm(word_embeds + self.pos_embedding[:word_embeds.size(2)] + self.token_type_embedding))
        embedding = self.dropOut(self.layerNorm(word_embeds))

        return embedding