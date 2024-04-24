from utils import *
from torch import nn

class DQNNet(nn.Module):
    def __init__(self):
        super(DQNNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(510, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 300)
        )

    def forward(self, x):
        return self.net(x)


class CrossNet(nn.Module):
    def __init__(self, embed_size, atten_size, prefer_size, contrib_size, n_actions):
        super(CrossNet, self).__init__()
        self.embed_layer = nn.Embedding(21, embed_size, padding_idx=20)

        self.embed_head = nn.Sequential(
            nn.Linear(embed_size * 49, 1024),
            nn.ReLU(),
            nn.Linear(1024, 520)
        )

        self.atten_head = nn.Sequential(
            nn.Linear(atten_size, 5680), # 512
            nn.ReLU(),
            nn.Linear(5680, 1380),
        )

        self.prefer_head = nn.Sequential(
            nn.Linear(prefer_size, 2560), # 300
            nn.ReLU(),
            nn.Linear(2560, 940),
            
        )

        self.contrib_head = nn.Sequential(
            nn.Linear(contrib_size, 2560),
            nn.ReLU(),
            nn.Linear(2560, 940),
        )

        self.combine_net = nn.Sequential(
            nn.Linear(3780, 3780),
            # nn.BatchNorm1d(3780),
            nn.ReLU(),
            nn.Linear(3780, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, n_actions)   
        )

    def forward(self, embed_batch, atten_batch, prefer_batch, contrib_batch):
        embed_batch = self.embed_layer(embed_batch)
        embed_batch = nn.Flatten()(embed_batch)
        embed_batch = self.embed_head(embed_batch)

        atten_batch = self.atten_head(atten_batch)
        prefer_batch = self.prefer_head(prefer_batch)
        contrib_batch = self.contrib_head(contrib_batch)
        combine_batch = torch.cat((embed_batch, atten_batch, prefer_batch, contrib_batch), dim=1)
        return self.combine_net(combine_batch)


class ValueNet(nn.Module):
    def __init__(self, embed_size, atten_size, prefer_size, contrib_size):
        super(ValueNet, self).__init__()
        self.embed_layer = nn.Embedding(21, embed_size, padding_idx=20)

        self.embed_head = nn.Sequential(
            nn.Linear(embed_size * 49, 1024),
            nn.ReLU(),
            nn.Linear(1024, 520)
        )

        self.atten_head = nn.Sequential(
            nn.Linear(atten_size, 5680), # 512
            nn.ReLU(),
            nn.Linear(5680, 1380),
        )

        self.prefer_head = nn.Sequential(
            nn.Linear(prefer_size, 2560), # 300
            nn.ReLU(),
            nn.Linear(2560, 940),
            
        )

        self.contrib_head = nn.Sequential(
            nn.Linear(contrib_size, 2560),
            nn.ReLU(),
            nn.Linear(2560, 940),
        )
    
        self.combine_net = nn.Sequential(
            nn.Linear(3780, 3780),
            nn.ReLU(),
            nn.Linear(3780, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )

    def forward(self, embed_batch, atten_batch, prefer_batch, contrib_batch):
        embed_batch = self.embed_layer(embed_batch)
        embed_batch = nn.Flatten()(embed_batch)
        embed_batch = self.embed_head(embed_batch)

        atten_batch = self.atten_head(atten_batch)
        prefer_batch = self.prefer_head(prefer_batch)
        contrib_batch = self.contrib_head(contrib_batch)
        combine_batch = torch.cat((embed_batch, atten_batch, prefer_batch, contrib_batch), dim=1)
        return self.combine_net(combine_batch)
