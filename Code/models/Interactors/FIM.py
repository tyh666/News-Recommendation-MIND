import math
import torch
import torch.nn as nn

# FIXME: no enough dimension for 3dcnn
# class FIM_Interactor(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.name = 'fim'

#         if config.k > 9:
#             self.final_dim = int(int(config.k / 3) /3) * int(int(config.title_length / 3) / 3)**2 * 16
#         else:
#             self.final_dim = (config.k-4) * int(int(config.title_length / 3) / 3)**2 * 16

#         self.SeqCNN3D = nn.Sequential(
#             nn.Conv3d(in_channels=config.level, out_channels=32, kernel_size=[3, 3, 3], padding=1),
#             nn.ReLU(),
#             nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[3, 3, 3]),
#             nn.Conv3d(in_channels=32, out_channels=16, kernel_size=[3, 3, 3], padding=1),
#             nn.ReLU(),
#             nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[3, 3, 3])
#         )

#         nn.init.xavier_normal_(self.SeqCNN3D[0].weight)
#         nn.init.xavier_normal_(self.SeqCNN3D[3].weight)


#     def forward(self, cdd_news_embedding, ps_terms, **kwargs):
#         """ compute interaction score between candidate news and personalized terms

#         Args:
#             cdd_news_embedding: [batch_size, cdd_size, signal_length, level, hidden_dim]
#             ps_terms: [batch_size, term_num, level, hidden_dim]

#         Returns:
#             matching_tensor: tensor of [batch_size, cdd_size, final_dim], where final_dim is derived from MaxPooling with no padding
#         """
#         cdd_news_embedding = cdd_news_embedding.transpose(-2, -3)
#         # [bs, 1, lv, tn, hd]
#         ps_terms = ps_terms.unsqueeze(1).transpose(-2, -3)

#         # [bs, cs, lv, sl, tn]
#         matching_tensor = cdd_news_embedding.matmul(ps_terms.transpose(-2, -1)) / math.sqrt(cdd_news_embedding.shape[-1])

#         # reshape the tensor in order to feed into 3D CNN pipeline
#         matching_tensor = matching_tensor.view(-1, matching_tensor.shape[2:])

#         matching_tensor = self.SeqCNN3D(matching_tensor).view(cdd_news_embedding.shape[0], cdd_news_embedding.shape[1], self.final_dim)
#         return matching_tensor


class FIM_Interactor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = 'fim'

        if config.k > 9:
            self.final_dim = int(int(config.k / 3) /3) * int(int(config.title_length / 3) / 3)**2 * 16
        else:
            self.final_dim = (config.k-4) * int(int(config.title_length / 3) / 3)**2 * 16

        if config.k > 9:
            self.SeqCNN3D = nn.Sequential(
                nn.Conv3d(in_channels=config.level, out_channels=32, kernel_size=[3, 3, 3], padding=1),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[3, 3, 3]),
                nn.Conv3d(in_channels=32, out_channels=16, kernel_size=[3, 3, 3], padding=1),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[3, 3, 3])
            )
        else:
            self.SeqCNN3D = nn.Sequential(
                nn.Conv3d(in_channels=config.level, out_channels=32, kernel_size=[3, 3, 3], padding=1),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[1, 3, 3]),
                nn.Conv3d(in_channels=32, out_channels=16, kernel_size=[3, 3, 3], padding=1),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[1, 3, 3])
            )
        nn.init.xavier_normal_(self.SeqCNN3D[0].weight)
        nn.init.xavier_normal_(self.SeqCNN3D[3].weight)


    def forward(self, cdd_news_embedding, his_activated, **kwargs):
        """ construct fusion tensor between candidate news repr and history news repr at each dilation level

        Args:
            cdd_news_embedding: tensor of [batch_size, cdd_size, signal_length, level, hidden_dim]
            his_activated: tensor of [batch_size, cdd_size, k, signal_length, level, hidden_dim]

        Returns:
            fusion_tensor: tensor of [batch_size, cdd_size, final_dim], where final_dim is derived from MaxPooling with no padding
        """
        cdd_news_embedding = cdd_news_embedding.transpose(-2, -3)
        # [bs, cs, k, lv, sl, hd]
        his_news_embedding = his_activated.transpose(-2, -3)

        # [batch_size, cdd_size, k, level, signal_length, signal_length]
        fusion_tensor = torch.matmul(cdd_news_embedding.unsqueeze(
            dim=2), his_news_embedding.transpose(-2, -1)) / math.sqrt(cdd_news_embedding.shape[-1])

        # reshape the tensor in order to feed into 3D CNN pipeline
        fusion_tensor = fusion_tensor.view(-1, his_news_embedding.shape[2], his_news_embedding.shape[3],
                                           his_news_embedding.shape[-2], his_news_embedding.shape[-2]).transpose(1, 2)

        fusion_tensor = self.SeqCNN3D(fusion_tensor).view(cdd_news_embedding.shape[0], cdd_news_embedding.shape[1], -1)
        return fusion_tensor