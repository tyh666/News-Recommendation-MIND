import torch.nn as nn

class Random_Embedding(nn.Module):
    """
        pretrained glove embedding
    """
    def __init__(self, manager, vocab):
        super().__init__()
        self.name = 'random'

        self.embedding_dim = manager.embedding_dim

        self.embedding = nn.Embedding(vocab)
            sparse=manager.spadam,
            freeze=False
        )

        self.dropOut = nn.Dropout(manager.dropout_p)

    def forward(self, news_batch):
        """
        Args:
            news_batch: [batch_size, *, signal_length]

        Returns:
            news_embedding: [batch_size, *, signal_length, embedding_dim]
        """
        embedding = self.dropOut(self.embedding(news_batch))
        return embedding