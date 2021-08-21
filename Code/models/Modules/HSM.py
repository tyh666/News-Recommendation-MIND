import torch
import torch.nn as nn
import torch.nn.functional as F


class SFI_Selector(nn.Module):
    """
    select most informative k history
    """
    def __init__(self, config):
        super().__init__()
        self.k = config.k
        self.his_size = config.his_size

        self.signal_length = config.signal_length
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim

        self.selectionProject = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        if config.threshold != -float('inf'):
            threshold = torch.tensor([config.threshold])
            self.register_buffer('threshold', threshold)

        for param in self.selectionProject:
            if isinstance(param, nn.Linear):
                nn.init.xavier_normal_(param.weight)

    def forward(self, cdd_repr, his_repr, his_embedding, his_attn_mask):
        """ apply news-level attention

        Args:
            cdd_repr: tensor of [batch_size, cdd_size, hidden_dim]
            his_repr: tensor of [batch_size, his_size, hidden_dim]
            his_embedding: tensor of [batch_size, his_size, signal_length, embedding_dim]
            his_attn_mask: tensor of [batch_size, his_size, signal_length]

        Returns:
            his_selected: tensor of [batch_size, cdd_size, k, signal_length, hidden_dim]
            his_mask_selected: tensor of [batch_size, cdd_size, k, signal_length]
        """

        # [bs, cs, hs]
        # t1 = time.time()
        batch_size = cdd_repr.size(0)
        cdd_size = cdd_repr.size(1)

        cdd_repr = F.normalize(self.selectionProject(cdd_repr),dim=-1)
        his_repr = F.normalize(self.selectionProject(his_repr),dim=-1)
        attn_weights = cdd_repr.matmul(his_repr.transpose(-1, -2))

        if self.k == self.his_size:

            his_selected = his_embedding.unsqueeze(1)
            his_mask_selected = his_attn_mask.unsqueeze(1)

        else:
            # t2 = time.time()
            # attn_weights = attn_weights.masked_fill(his_mask.transpose(-1, -2), -float("inf"))
            attn_weights, attn_weights_index = attn_weights.topk(dim=-1, k=self.k)

            # print(attn_weights, attn_weights_index)
            # t3 = time.time()

            # [bs, cs, k, sl, level, fn]
            his_selected = his_embedding.unsqueeze(dim=1).expand(batch_size, cdd_size, self.his_size, self.signal_length, self.embedding_dim).gather(
                dim=2,
                index=attn_weights_index.view(batch_size, cdd_size, self.k, 1, 1).expand(batch_size, cdd_size, self.k, self.signal_length, self.embedding_dim)
            )

            his_mask_selected = his_attn_mask.unsqueeze(dim=1).expand(batch_size, cdd_size, self.his_size, self.signal_length).gather(
                dim=2,
                index=attn_weights_index.view(batch_size, cdd_size, self.k, 1).expand(batch_size, cdd_size, self.k, self.signal_length)
            )

            # t4 = time.time()

        if hasattr(self,'threshold'):
            his_selected = his_selected * (attn_weights.masked_fill(attn_weights<self.threshold, 0).view(batch_size, cdd_size, self.k, 1, 1))
            # his_selected = his_selected * (F.softmax(attn_weights.masked_fill(attn_weights<self.threshold, 0), dim=-1).view(batch_size, self.cdd_size, self.k, 1, 1, 1))

        # t6 = time.time()
        # print("product time:{}, sort time:{}, scatter time:{}, activate time:{}, mask time:{}".format(t2-t1, t3-t2, t4-t3, t5-t4, t6-t5))

        return his_selected, his_mask_selected


class Recent_Selector(nn.Module):
    """
    select recent k history
    """
    def __init__(self, config):
        super().__init__()
        self.k = config.k
        self.his_size = config.his_size
        self.signal_length = config.signal_length

    def forward(self, cdd_repr, his_repr, his_embedding, his_attn_mask):
        """ apply news-level attention

        Args:
            cdd_repr: tensor of [batch_size, cdd_size, hidden_dim]
            his_repr: tensor of [batch_size, his_size, hidden_dim]
            his_embedding: tensor of [batch_size, his_size, signal_length, embedding_dim]
            his_attn_mask: tensor of [batch_size, his_size, signal_length]

        Returns:
            his_selected: tensor of [batch_size, cdd_size, k, signal_length, embedding_dim]
            his_mask_selected: tensor of [batch_size, cdd_size, k, signal_length]
        """
        cdd_size = cdd_repr.size(1)

        his_selected = his_embedding[:, :self.k].unsqueeze(1).repeat(1, cdd_size, 1, 1, 1)
        his_mask_selected = his_attn_mask[:, :self.k].unsqueeze(1).repeat(1, cdd_size, 1, 1)

        return his_selected, his_mask_selected