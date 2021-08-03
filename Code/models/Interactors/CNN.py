import math
import torch.nn as nn

class CNN_Interactor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = '2dcnn'

        self.term_num = config.his_size * config.k
        self.level = config.level
        self.hidden_dim = config.hidden_dim

        self.signal_length = config.signal_length

        self.SeqCNN2D = nn.Sequential(
            nn.Conv2d(in_channels=self.level, out_channels=32, kernel_size=[3, 3], padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[3, 3], stride=[3, 3]),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=[3, 3], padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[3, 3], stride=[3, 3])
        )

        self.final_dim = int(int(self.signal_length/3)/3) * int(int(self.term_num/3)/3) * 16

        nn.init.xavier_normal_(self.SeqCNN2D[0].weight)
        nn.init.xavier_normal_(self.SeqCNN2D[3].weight)

    def forward(self, cdd_news_embedding, ps_terms):
        """
        calculate interaction tensor and reduce it to a vector

        Args:
            cdd_news_embedding: word-level representation of candidate news, [batch_size, cdd_size, signal_length, level, hidden_dim]
            ps_terms: personalized terms, [batch_size, his_size, k, level, hidden_dim]

        Returns:
            reduced_tensor: output tensor after CNN2d, [batch_size, cdd_size, final_dim]
        """
        batch_size = cdd_news_embedding.size(0)
        cdd_news_embedding = cdd_news_embedding.transpose(-2, -3)
        # [bs, 1, lv, tn, hd]
        ps_terms = ps_terms.view(batch_size, 1, -1, self.level, self.hidden_dim).transpose(-2, -3)

        # [bs, cs, lv, sl, tn]
        matching_tensor = cdd_news_embedding.matmul(ps_terms.transpose(-2, -1)) / math.sqrt(cdd_news_embedding.shape[-1])

        # reshape the tensor in order to feed into 3D CNN pipeline
        matching_tensor = matching_tensor.view((-1,) + matching_tensor.shape[2:])

        reduced_tensor = self.SeqCNN2D(matching_tensor).view(batch_size, cdd_news_embedding.size(1), self.final_dim)
        return reduced_tensor

if __name__ == '__main__':
    import torch
    a = torch.rand(2,2,10,2,5)
    b = torch.rand(2,5,2,2,5)
    drm = CNN_Interactor(10,10,2,5)
    c = drm(a,b)
    print(c, c.shape)