import torch
import torch.nn as nn
import torch.nn.functional as F


class SFI_Selector(nn.Module):
    """
    select most informative k history
    """
    def __init__(self, manager):
        super().__init__()
        self.k = manager.k
        self.his_size = manager.his_size

        self.signal_length = manager.signal_length
        self.embedding_dim = manager.embedding_dim
        self.hidden_dim = manager.hidden_dim

        self.selectionProject = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        if manager.threshold != -float('inf'):
            threshold = torch.tensor([manager.threshold])
            self.register_buffer('threshold', threshold)

        for param in self.selectionProject:
            if isinstance(param, nn.Linear):
                nn.init.xavier_normal_(param.weight)

        keep_k_modifier = torch.zeros(1, 1, manager.his_size)
        keep_k_modifier[:, :, :self.k] = 1
        self.register_buffer('keep_k_modifier', keep_k_modifier, persistent=False)

    def forward(self, cdd_repr, his_repr, his_embedding, his_attn_mask, his_mask):
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
            pad_pos = ~((his_mask.transpose(-1, -2) + self.keep_k_modifier).bool())
            attn_weights = attn_weights.masked_fill(pad_pos, -float("inf"))
            attn_weights, attn_weights_index = attn_weights.topk(dim=-1, k=self.k)

            # [bs, cs, k, sl, level, fn]
            his_selected = his_embedding.unsqueeze(dim=1).expand(batch_size, cdd_size, self.his_size, self.signal_length, self.embedding_dim).gather(
                dim=2,
                index=attn_weights_index.view(batch_size, cdd_size, self.k, 1, 1).expand(batch_size, cdd_size, self.k, self.signal_length, self.embedding_dim)
            )

            his_mask_selected = his_attn_mask.unsqueeze(dim=1).expand(batch_size, cdd_size, self.his_size, self.signal_length).gather(
                dim=2,
                index=attn_weights_index.view(batch_size, cdd_size, self.k, 1).expand(batch_size, cdd_size, self.k, self.signal_length)
            )

        if hasattr(self, 'threshold'):
            # bs, cs, k
            mask_pos = attn_weights < self.threshold
            # his_selected = his_selected * (attn_weights.masked_fill(mask_pos, 0).view(batch_size, cdd_size, self.k, 1, 1))
            his_selected = his_selected * (F.softmax(attn_weights.masked_fill(attn_weights<self.threshold, 0), dim=-1).view(batch_size, self.cdd_size, self.k, 1, 1, 1))
            his_mask_selected = his_mask_selected * (~mask_pos.unsqueeze(-1))

        return his_selected, his_mask_selected


class Recent_Selector(nn.Module):
    """
    select recent k history
    """
    def __init__(self, manager):
        super().__init__()
        self.k = manager.k
        self.his_size = manager.his_size
        self.signal_length = manager.signal_length

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