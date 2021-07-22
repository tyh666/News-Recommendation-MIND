import math
import torch.nn as nn

class CNN_Interactor(nn.Module):
    def __init__(self, signal_length, term_num, hidden_dim):
        super().__init__()
        self.name = '2dcnn'
        self.hidden_dim = hidden_dim
        self.signal_length = signal_length
        self.term_num = term_num

        self.SeqCNN2D = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=[3, 3], padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[3, 3], stride=[3, 3]),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=[3, 3], padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[3, 3], stride=[3, 3])
        )

        self.final_dim = int(int(signal_length/3)/3) * int(int(term_num/3)/3) * 16

        nn.init.xavier_normal_(self.SeqCNN2D[0].weight)
        nn.init.xavier_normal_(self.SeqCNN2D[3].weight)

    def forward(self, cdd_news_embedding, ps_terms):
        """
        calculate interaction tensor and reduce it to a vector

        Args:
            cdd_news_embedding: word-level representation of candidate news, [batch_size, cdd_size, signal_length, hidden_dim]
            ps_terms: personalized terms, [batch_size, term_num, hidden_dim]

        Returns:
            reduced_tensor: output tensor after CNN2d, [batch_size, cdd_size, final_dim]
        """

        # [bs, cs, sl, tn]
        matching_tensor = cdd_news_embedding.matmul(ps_terms.transpose(-2,-1).unsqueeze(1)).view(-1, 1, self.signal_length, self.term_num) / math.sqrt(self.hidden_dim)
        reduced_tensor = self.SeqCNN2D(matching_tensor).view(cdd_news_embedding.size(0), cdd_news_embedding.size(1), self.final_dim)
        return reduced_tensor