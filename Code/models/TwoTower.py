from .TwoTowerBaseModel import TwoTowerBaseModel

class TwoTower(TwoTowerBaseModel):
    """
    Tow tower model
    """
    def __init__(self, manager, embedding, encoderN, encoderU):
        super().__init__(manager)

        self.embedding = embedding
        self.encoderN = encoderN
        self.encoderU = encoderU

        self.hidden_dim = manager.hidden_dim

        manager.name = '__'.join(['twotower', manager.encoderN, manager.encoderU])
        # used in fast evaluate
        self.name = manager.name


    def encode_news(self, x):
        """
        encode candidate news
        """
        # encode news with MIND_news
        cdd_news = x["cdd_encoded_index"].to(self.device)
        cdd_attn_mask = x['cdd_attn_mask'].to(self.device)

        _, cdd_news_repr = self.encoderN(
            self.embedding(cdd_news), cdd_attn_mask
        )

        return cdd_news_repr


    def encode_user(self, x):
        """
        encoder user
        """
        his_news = x["his_encoded_index"].to(self.device)
        his_attn_mask = x["his_attn_mask"].to(self.device)

        _, his_news_repr = self.encoderN(
            self.embedding(his_news), his_attn_mask
        )

        user_repr = self.encoderU(his_news_repr, his_mask=x['his_mask'], user_id=x['user_id'].to(self.device))

        return user_repr, None