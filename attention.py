import torch
import math
from torch import nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_device():
    return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def scaled_dot_product(q, k, v, mask=None):
    # q: 30 x 8 x 200 x 64, k: 30 x 8 x 200 x 64, v: 30 x 8 x 200 x 64, mask 200 x 200
    d_k = q.size()[-1] 

    scaled = torch.matmul(q, k.transpose(-1, -2)).to(device='cuda:0') / math.sqrt(d_k) # 30 x 8 x 200 x 200
    scaled = scaled.to(device='cpu')
    torch.cuda.empty_cache()
    print(f"scaled.size() : {scaled.size()}")

    if mask is not None:
        print(f"-- ADDING MASK of shape {mask.size()} --") 
        scaled += mask # 30 x 8 x 200 x 200
    attention = F.softmax(scaled, dim=-1).to(device='cuda:0')
     # 30 x 8 x 200 x 200
    values = torch.matmul(attention, v.to(device='cuda:0')).to(device='cuda:0') # 30 x 8 x 200 x 64
    print('values in ', values.device)
    values=values.to('cpu')
    torch.cuda.empty_cache()
    return values, attention

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model , 3 * d_model).to(get_device()) # 1536 
        self.linear_layer = nn.Linear(d_model, d_model).to(get_device())
    
    def forward(self, x, mask=None):
        batch_size, sequence_length, d_model = x.size() # 30 x 200 x 512 
        #print(f"x.size(): {x.size()}")
        qkv = self.qkv_layer(x.to(device='cuda:0'))
        qkv = qkv.to(device='cpu')
        torch.cuda.empty_cache() 
        print('qkv in',qkv.device)
        #print(f"qkv.size(): {qkv.size()}")

        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim).to(device='cuda:0') # 30 x 200 x 8 x 192
        qkv = qkv.to(device='cpu')
        torch.cuda.empty_cache() 
        print('qkv in',qkv.device)
        #print(f"qkv after reshape .size(): {qkv.size()}")

        qkv = qkv.permute(0, 2, 1, 3).to(device='cuda:0') # 30 x 8 x 200 x 192
        qkv = qkv.to(device='cpu')
        torch.cuda.empty_cache() 
        print('qkv in',qkv.device)
        #print(f"qkv after permutation: {qkv.size()}")

        q, k, v = qkv.chunk(3, dim=-1) # q: 30 x 8 x 200 x 64, k: 30 x 8 x 200 x 64, v: 30 x 8 x 200 x 64
        
        #print(f"q: {q.size()}, k:{k.size()}, v:{v.size()}")

        values, attention = scaled_dot_product(q, k, v, mask) # values: 30 x 8 x 200 x 64
        #print(f"values: {values.size()}, attention:{attention.size()}")
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim) # 30 x 200 x 512
        #print(f"values after reshaping: {values.size()}")
        out = self.linear_layer(values.to(device='cuda:0')) # 30 x 200 x 512
        out = out.to(device='cpu')
        torch.cuda.empty_cache()
        print(f"Multi headed hattention done , size=: {out.size()}",'\n')
        return out # 30 x 200 x 512
    

    
class MultiHeadCrossAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model , 2 * d_model).to(get_device())# 1024
        self.q_layer = nn.Linear(d_model , d_model).to(get_device())
        self.linear_layer = nn.Linear(d_model, d_model).to(get_device())
    
    def forward(self, x, y, mask=None):
        batch_size, sequence_length, d_model = x.size() # 30 x 200 x 512
        #print(f"x.size(): {x.size()}")
        kv = self.kv_layer(x.to(device='cuda:0')) # 30 x 200 x 1024
        #print(f"kv.size(): {kv.size()}")
        q = self.q_layer(y.to(device='cuda:0')) # 30 x 200 x 512
        #print(f"q.size(): {q.size()}")
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)  # 30 x 200 x 8 x 128
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)  # 30 x 200 x 8 x 64
        kv = kv.permute(0, 2, 1, 3) # 30 x 8 x 200 x 128
        q = q.permute(0, 2, 1, 3) # 30 x 8 x 200 x 64
        k, v = kv.chunk(2, dim=-1) # K: 30 x 8 x 200 x 64, v: 30 x 8 x 200 x 64
        values, attention = scaled_dot_product(q, k, v, mask) #  30 x 8 x 200 x 64
        #print(f"values: {values.size()}, attention:{attention.size()}")
        values = values.reshape(batch_size, sequence_length, d_model).to(device='cuda:0') #  30 x 200 x 512
        values = values.to('cpu')
        torch.cuda.empty_cache()
        out = self.linear_layer(values.to(device='cuda:0'))
        out = out.to(device='cpu')  #  30 x 200 x 512
        torch.cuda.empty_cache()
        print(f"Cross attention completed size=: {out.size()}",'\n')
        return out  #  30 x 200 x 512


x = torch.randn( (10000,104,128) ) # input positional encoded
y = torch.randn( (10000,104,128) ) # output sentence positional encoded 
mask = torch.full([104,104] , float('-inf'))
mask = torch.triu(mask, diagonal=1) # Mask initialization for masked attention


A = MultiHeadCrossAttention(128,4)
for i in range(10):
    print(f'Attention {i}')
    A.forward(x,y,mask)