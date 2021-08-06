import torch.nn as nn


class GLOVE_Embedding(nn.Module):
    """
        pretrained glove embedding
    """
    def __init__(self, config, vocab):
        super().__init__()
        self.name = 'glove'

        self.embedding_dim = 300
        config.embedding_dim = self.embedding_dim

        # FIXME: multiview
        config.signal_length = config.title_length

        self.embedding = nn.Embedding.from_pretrained(
            vocab.vectors,
            sparse=config.spadam,
            freeze=False
        )

        self.dropOut = nn.Dropout(config.dropout_p)

    def forward(self, news_batch):
        """
        Args:
            news_batch: [batch_size, *, signal_length]

        Returns:
            news_embedding: [batch_size, *, signal_length, embedding_dim]
        """
        embedding = self.dropOut(self.embedding(news_batch))
        return embedding