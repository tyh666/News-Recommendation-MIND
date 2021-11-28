import torch.nn as nn
from transformers import AutoModel
from models.UniLM.configuration_tnlrv3 import TuringNLRv3Config
from models.UniLM.modeling import TuringNLRv3ForSequenceClassification, relative_position_bucket

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

        if manager.bert == 'unilm':
            config = TuringNLRv3Config.from_pretrained(manager.unilm_config_path)
            # config.pooler = None
            bert = TuringNLRv3ForSequenceClassification.from_pretrained(manager.unilm_path, config=config).bert

        else:
            bert = AutoModel.from_pretrained(
                manager.get_bert_for_load(),
                cache_dir=manager.path + 'bert_cache/'
            )

        self.bert_word_embedding = bert.embeddings.word_embeddings

        if manager.reducer == 'bow':
            self.freq_embedding = nn.Embedding(manager.signal_length // 2, self.hidden_dim)
            nn.init.xavier_normal_(self.freq_embedding.weight)

    def forward(self, news_batch, subword_prefix=None):
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

        # word-level
        if subword_prefix is not None:
            word_embeds = subword_prefix.matmul(word_embeds)

        return word_embeds