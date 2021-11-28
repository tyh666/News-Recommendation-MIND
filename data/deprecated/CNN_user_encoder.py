class CNN_User_Encoder(nn.Module):
    def __init__(self, manager):
        super().__init__()

        self.hidden_dim = manager.hidden_dim
        self.SeqCNN1D = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim//2, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool1d(3,3),
            nn.Conv1d(self.hidden_dim//2, self.hidden_dim//2//2, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool1d(3,3)
        )
        self.userProject = nn.Linear((self.hidden_dim // 2 // 2) * (manager.his_size // 3 // 3), self.hidden_dim)

        nn.init.xavier_normal_(self.SeqCNN1D[0].weight)
        nn.init.xavier_normal_(self.SeqCNN1D[3].weight)
        nn.init.xavier_normal_(self.userProject.weight)

    def forward(self, news_reprs):
        """
        encode user history into a representation vector by 1D CNN and Max Pooling

        Args:
            news_reprs: batch of news representations, [batch_size, *, hidden_dim]

        Returns:
            user_repr: user representation (coarse), [batch_size, 1, hidden_dim]
        """
        batch_size = news_reprs.size(0)
        encoded_reprs = self.SeqCNN1D(news_reprs.transpose(-2,-1)).view(batch_size, -1)
        user_repr = self.userProject(encoded_reprs).unsqueeze(1)
        return user_repr