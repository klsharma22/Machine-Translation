import torch
import math
from torch import nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden).to(device)
        self.linear2 = nn.Linear(hidden, d_model).to(device)
        self.relu = nn.ReLU().to(device)
        self.dropout = nn.Dropout(p=drop_prob).to(device)

    def forward(self, x):
        x = self.linear1(x.to(device))
        #print(f"x after first linear layer: {x.size()}")
        x = self.relu(x)
        #print(f"x after activation: {x.size()}")
        x = self.dropout(x)
        #print(f"x after dropout: {x.size()}")
        x = self.linear2(x)
        x = x.to(device='cpu')
        torch.cuda.empty_cache()
        print(f"x feed forward network: {x.size()}")
        return x

x = torch.randn( (30000,104,128) )
P = PositionwiseFeedForward(128,256)
for i in range(10):
    print(f'FFN {i}')
    P.forward(x)