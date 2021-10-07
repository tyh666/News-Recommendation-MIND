import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, manager):
        super().__init__()

        self.scale = manager.scale
        self.cdd_size = manager.cdd_size

        self.batch_size = manager.batch_size
        # used for fast evaluate
        self.batch_size_news = manager.batch_size_news
        self.batch_size_history = manager.batch_size_history
        # not enable fast evaluation mode
        self.ready_encode = False

        self.his_size = manager.his_size
        self.signal_length = manager.signal_length
        self.device = manager.device


    def _init_embedding(self):
        """
        prepare for fast inferring
        """
        self.cache_directory = "data/cache/{}/{}/".format(self.name, self.scale)
        self.news_reprs = nn.Embedding.from_pretrained(torch.load(self.cache_directory + "news.pt", map_location=torch.device(self.device)))


    def _init_encoding(self):
        """
        prepare for fast encoding
        """
        if self.granularity != 'token':
            self.cdd_dest = torch.zeros((self.batch_size_news, self.signal_length * self.signal_length), device=self.device)
            # if self.reducer in ["bm25", "entity", "first"]:
            #     self.his_dest = torch.zeros((self.batch_size_history, self.his_size, (self.k + 1) * (self.k + 1)), device=self.device)
            # else:
            #     self.his_dest = torch.zeros((self.batch_size_history, self.his_size, self.signal_length * self.signal_length), device=self.device)
        self.ready_encode = True