import os
import sys
os.chdir('./')
sys.path.append('./')

import torch
import torch.nn as nn
from models.base_model import BaseModel

from models.Encoders.CNN import CNN_Encoder
from models.Encoders.RNN import RNN_User_Encoder

class CNN_RNN_Dot(BaseModel):
    def __init__(self, config, vocab):
        super().__init__(config)
        self.title_size = config.title_size
        self.k = config.k

        self.encoderN = CNN_Encoder(config, vocab)
        self.encoderU = RNN_User_Encoder(self.encoderN.hidden_dim)

        self.hidden_dim = self.encoderN.hidden_dim

        self.name = '-'.join(['baseline', self.encoderN.name, self.encoderU.name, 'dot'])

    def clickPredictor(self, user_repr, cdd_news_repr):
        """ calculate batch of click probabolity

        Args:
            user_repr: [batch_size, 1, hiddedn_dim]
            cdd_news_repr: [batch_size, cdd_size, hidden_dim]

        Returns:
            score of each candidate news, [batch_size, cdd_size]
        """
        return cdd_news_repr.matmul(user_repr.transpose(-1,-2)).squeeze(-1)

    def _forward(self,x):
        if x['cdd_encoded_index'].size(0) != self.batch_size:
            self.batch_size = x['cdd_encoded_index'].size(0)

        cdd_news = x['cdd_encoded_index'].long().to(self.device)
        cdd_news_embedding, cdd_news_repr = self.encoderN(
            cdd_news)
        his_news = x["his_encoded_index"].long().to(self.device)
        his_news_embedding, his_news_repr = self.encoderN(
            his_news)

        user_repr = self.encoderU(his_news_repr)

        return self.clickPredictor(user_repr, cdd_news_repr)

    def forward(self,x):
        """
        Decoupled function, score is unormalized click score
        """
        score = self._forward(x)

        if self.cdd_size > 1:
            prob = nn.functional.log_softmax(score, dim=1)
        else:
            prob = torch.sigmoid(score)

        return prob

if __name__ == '__main__':
    from utils.utils import prepare, load_config
    from data.configs.drm import config

    config = load_config(config)
    vocab, loaders = prepare(config)
    baseline = CNN_RNN_Dot(config, vocab).to(config.device)

    if config.mode == 'dev':
        baseline.evaluate(config,loaders[0],loading=True)

    elif config.mode == 'train':
        baseline.fit(config, loaders)

    elif config.mode == 'tune':
        baseline.tune(config, loaders)

    elif config.mode == 'test':
        baseline.test(config, loaders[0])