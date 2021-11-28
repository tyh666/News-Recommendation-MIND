import math
import torch.nn as nn

class CNN_Ranker(nn.Module):
    def __init__(self, manager):
        super().__init__()
        self.name = '2dcnn'

        self.term_num = manager.term_num
        self.embedding_dim = manager.embedding_dim

        self.signal_length = manager.signal_length

        self.SeqCNN2D = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=[3, 3], padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[3, 3], stride=[3, 3]),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=[3, 3], padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[3, 3], stride=[3, 3])
        )

        self.final_dim = int(int(self.signal_length/3)/3) * int(int(self.term_num/3)/3) * 16

        nn.init.xavier_normal_(self.SeqCNN2D[0].weight)
        nn.init.xavier_normal_(self.SeqCNN2D[3].weight)

    def forward(self, cdd_news_embedding, ps_terms, *args):
        """
        calculate interaction tensor and reduce it to a vector

        Args:
            ps_terms: personalized terms, [batch_size, his_size, k, hidden_dim]
            cdd_news_embedding: word-level representation of candidate news, [batch_size, cdd_size, signal_length, hidden_dim]

        Returns:
            reduced_tensor: output tensor after CNN2d, [batch_size, cdd_size, final_dim]
        """
        batch_size = cdd_news_embedding.size(0)
        cdd_size = cdd_news_embedding.size(1)
        # [bs, 1, tn, hd]
        ps_terms = ps_terms.view(batch_size, 1, self.term_num, -1)

        # [bs, cs, sl, tn]
        matching_tensor = cdd_news_embedding.matmul(ps_terms.transpose(-2, -1)) / math.sqrt(cdd_news_embedding.size(-1))

        # reshape the tensor in order to feed into 3D CNN pipeline
        matching_tensor = matching_tensor.view(-1, *matching_tensor.shape[2:]).unsqueeze(1)
        reduced_tensor = self.SeqCNN2D(matching_tensor).view(batch_size, cdd_size, self.final_dim)

        return reduced_tensor