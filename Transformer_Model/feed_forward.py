import torch
import math
from torch import nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        #print(f"x after first linear layer: {x.size()}")
        x = self.relu(x)
        #print(f"x after activation: {x.size()}")
        x = self.dropout(x)
        #print(f"x after dropout: {x.size()}")
        x = self.linear2(x)
        print(f"x feed forward network: {x.size()}")
        return x