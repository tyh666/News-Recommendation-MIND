import math
import torch.nn as nn

# FIXME: no enough dimension for 3dcnn
class FIM_Interactor(nn.Module):
    def __init__(self, level):
        super().__init__()
        self.name = 'fim'

        self.SeqCNN3D = nn.Sequential(
            nn.Conv3d(in_channels=level, out_channels=32, kernel_size=[3, 3, 3], padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[3, 3, 3]),
            nn.Conv3d(in_channels=32, out_channels=16, kernel_size=[3, 3, 3], padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[3, 3, 3])
        )

        nn.init.xavier_normal_(self.SeqCNN3D[0].weight)
        nn.init.xavier_normal_(self.SeqCNN3D[3].weight)


    def forward(self, cdd_news_embedding, ps_terms, **kwargs):
        """ compute interaction score between candidate news and personalized terms

        Args:
            cdd_news_embedding: [batch_size, cdd_size, signal_length, level, hidden_dim]
            ps_terms: [batch_size, term_num, level, hidden_dim]

        Returns:
            matching_tensor: tensor of [batch_size, cdd_size, final_dim], where final_dim is derived from MaxPooling with no padding
        """
        cdd_news_embedding = cdd_news_embedding.transpose(-2, -3)
        # [bs, 1, lv, tn, hd]
        ps_terms = ps_terms.unsqueeze(1).transpose(-2, -3)

        # [bs, cs, lv, sl, tn]
        matching_tensor = cdd_news_embedding.matmul(ps_terms.transpose(-2, -1)) / math.sqrt(cdd_news_embedding.shape[-1])

        # reshape the tensor in order to feed into 3D CNN pipeline
        matching_tensor = matching_tensor.view(-1, matching_tensor.shape[2:])

        matching_tensor = self.SeqCNN3D(matching_tensor).view(cdd_news_embedding.shape[0], cdd_news_embedding.shape[1], -1)
        return matching_tensor