import torch
import math
from torch import nn
import torch.nn.functional as F
     

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    #print(f"scaled.size() : {scaled.size()}")
    if mask is not None:
        print(f"-- ADDING MASK of shape {mask.size()} --") 
        # Broadcasting add. So just the last N dimensions need to match
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model                                # 128
        self.num_heads = num_heads                            # 4
        self.head_dim = d_model // num_heads                  # 128/4 = 32 
        self.qkv_layer = nn.Linear(d_model , 3 * d_model)     # 128 x 384
        self.linear_layer = nn.Linear(d_model, d_model)       # 128 x 128
    
    def forward(self, x, mask=None):
        batch_size, max_sequence_length, d_model = x.size()
        #print(f"x.size(): {x.size()}")                        # 30 x 104 x 128
        qkv = self.qkv_layer(x)  
        #print(f"qkv.size(): {qkv.size()}")                    # 30 x 104 x 384
        qkv = qkv.reshape(batch_size, max_sequence_length, self.num_heads, 3 * self.head_dim)
        #print(f"qkv.size(): {qkv.size()}")                    # 30 x 104 x 4 x 96
        qkv = qkv.permute(0, 2, 1, 3)                         
        #print(f"qkv.size(): {qkv.size()}")                    # 30 x 4 x 104 x 96
        q, k, v = qkv.chunk(3, dim=-1)
        #print(f"q size: {q.size()}, k size: {k.size()}, v size: {v.size()}, ") # 3 tensors of [30 x 4 x 104 x 32] dimensions
        values, attention = scaled_dot_product(q, k, v, mask) # [30 x 4 x 104 x 32]
        #print(f"values.size(): {values.size()}, attention.size:{ attention.size()} ")
        values = values.reshape(batch_size, max_sequence_length, self.num_heads * self.head_dim)
        #print(f"values.size(): {values.size()}") #  [30 x 104 x 128]
        out = self.linear_layer(values)
        print(f"output of attention.size(): {out.size()}",'\n')
        return out #[30 x 104 x 32]


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta =  nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        #print(f"Mean ({mean.size()})")
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        #print(f"Standard Deviation  ({std.size()})")
        y = (inputs - mean) / std
        #print(f"y: {y.size()}")
        out = self.gamma * y  + self.beta
        #print(f"self.gamma: {self.gamma.size()}, self.beta: {self.beta.size()}")
        print(f" LayerNormalization size: {out.size()}",'\n')
        return out

  
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

from attention import MultiHeadAttention,MultiHeadCrossAttention
from layer_normalization import LayerNormalization
from feed_forward import PositionwiseFeedForward
class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob,i):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.i_th_encoder = i+1

    def forward(self, x):
        print(f'------ ENCODER LAYER NUMBER {self.i_th_encoder}----------')
        residual_x = x
        print("ATTENTION 1",)
        x = self.attention.forward(x, mask=None)
        print("DROPOUT 1",'\n')
        x = self.dropout1(x)
        print("-ADD AND LAYER NORMALIZATION 1 -")
        x = self.norm1(x + residual_x)
        residual_x = x
        print(" FEED FORWARD NETWORK")
        x = self.ffn(x)
        print("-DROPOUT 2 -",'\n')
        x = self.dropout2(x)
        print("-ADD AND LAYER NORMALIZATION 2-")
        x = self.norm2(x + residual_x)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob,i)
                                     for i in range(num_layers)])

    def forward(self, x):
        x = self.layers(x)
        return x
     




d_model = 128
num_heads = 4
drop_prob = 0.1
batch_size = 30
max_sequence_length = 104
ffn_hidden = 2048
num_layers = 5
#encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
     

#x = torch.randn( (batch_size, max_sequence_length, d_model) ) # includes positional encoding [30 x 104 x 128]
#out = encoder.forward(x)

