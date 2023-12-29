import torch
import math
from torch import nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape):
        super().__init__()
        self.parameters_shape=parameters_shape
        self.norm = nn.LayerNorm(self.parameters_shape).to(device)

    def forward(self, inputs):
        inputs = inputs.to(device)
        out = self.norm(inputs)
        out = out.to(device)
        torch.cuda.empty_cache()
        print(f"Layer normalization done , Size=: {out.size()}",'\n')
        return out

A = LayerNormalization(128)
x = torch.randn( (30000,104,128) )
for i in range(10):
    print(f'LN {i}')
    A.forward(x)