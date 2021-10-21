import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, manager):
        super().__init__()

        self.scale = manager.scale
        self.cdd_size = manager.cdd_size
        self.mode = manager.get_mode_for_cache()

        self.impr_size = manager.impr_size
        self.batch_size = manager.batch_size
        # used for encoding
        self.batch_size_news = manager.batch_size_news
        # encoding flag set to false
        self.encoding = False

        self.his_size = manager.his_size
        self.signal_length = manager.signal_length
        self.device = manager.device


    def init_encoding(self):
        """
        prepare for fast encoding
        """
        if self.granularity != 'token':
            self.cdd_dest = torch.zeros((self.batch_size_news, self.signal_length * self.signal_length), device=self.device)
        self.encoding = True


    def init_embedding(self):
        """
        prepare for fast inferring
        """
        self.cache_directory = "data/cache/{}/{}/{}/".format(self.name, self.scale, self.mode)
        self.news_reprs = nn.Embedding.from_pretrained(torch.load(self.cache_directory + "news.pt", map_location=torch.device(self.device)))


    def destroy_encoding(self):
        if self.granularity != 'token':
            self.cdd_dest = torch.zeros((self.batch_size, self.impr_size, self.signal_length * self.signal_length), device=self.device)
        self.encoding = False


    def destroy_embedding(self):
        self.news_reprs = None
        del self.news_reprs