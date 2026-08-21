import torch
import torch.nn as nn


class SnakeNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(11, 128),

            nn.ReLU(),

            nn.Linear(128, 128),

            nn.ReLU(),

            nn.Linear(128, 3)
        )

    def forward(self, x):

        return self.network(x)