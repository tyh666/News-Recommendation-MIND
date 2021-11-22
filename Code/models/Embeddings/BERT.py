import torch.nn as nn
from transformers import AutoModel

class BERT_Embedding(nn.Module):
    """
        1. convert token id to its embedding vector
        2. convert token frequency to its embedding if using bag-of-words
        3. slice/average/summarize subword embedding into the word embedding
        4. apply layerNorm and dropOut
    """
    def __init__(self, manager):
        super().__init__()

        self.hidden_dim = manager.bert_dim

        bert = AutoModel.from_pretrained(
            manager.get_bert_for_load(),
            cache_dir=manager.path + 'bert_cache/'
        )

        self.bert_word_embedding = bert.embeddings.word_embeddings


    def forward(self, news_batch):
        """ encode news with bert

        Args:
            news_batch: batch of news tokens, of size [batch_size, *, signal_length]
            bow: whether the input is bow

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, emedding_dim]
        """
        # bag of words
        if news_batch.dim() == 4:
            word_embeds = self.bert_word_embedding(news_batch[:,:,:,0]) + self.freq_embedding(news_batch[:,:,:,1])

        else:
            word_embeds = self.bert_word_embedding(news_batch)

        return word_embeds