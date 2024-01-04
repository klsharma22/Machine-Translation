from attention import MultiHeadAttention,MultiHeadCrossAttention
from layer_normalization import LayerNormalization
from feed_forward import PositionwiseFeedForward
import torch
import math
from torch import nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    


class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob,i):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.cross_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)
        self.i_th_decoder = i+1

    def forward(self, x, y, decoder_mask):
        print(f'------- DEOCDER LAYER NUMBER {self.i_th_decoder}-----------','\n')
        _y = y # 30 x 200 x 512
        print("MASKED SELF ATTENTION")
        y = self.self_attention(y, mask=decoder_mask) # 30 x 200 x 512
        print("DROP OUT 1")
        y = self.dropout1(y) # 30 x 200 x 512
        print("ADD + LAYER NORMALIZATION 1")
        y = self.norm1(y + _y) # 30 x 200 x 512

        _y = y # 30 x 200 x 512
        print("CROSS ATTENTION")
        y = self.cross_attention(x, y, mask=None) #30 x 200 x 512
        print("DROP OUT 2",'\n')  #30 x 200 x 512
        y = self.dropout2(y)
        print("ADD + LAYER NORMALIZATION 2")
        y = self.norm2(y + _y)  #30 x 200 x 512

        _y = y  #30 x 200 x 512
        print("FEED FORWARD 1")
        y = self.ffn(y) #30 x 200 x 512
        print("DROP OUT 3",'\n')
        y = self.dropout3(y) #30 x 200 x 512
        print("ADD + LAYER NORMALIZATION 3")
        y = self.norm3(y + _y) #30 x 200 x 512
        return y #30 x 200 x 512

class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, mask = inputs
        for module in self._modules.values():
            y = module(x, y, mask) #30 x 200 x 512
        return y

class Decoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers=1):
        super().__init__()
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob,i) 
                                          for i in range(num_layers)])

    def forward(self, x, y, mask):
        #x : 30 x 200 x 512 
        #y : 30 x 200 x 512
        #mask : 200 x 200
        y = self.layers(x, y, mask)
        return y #30 x 200 x 512
    

class Transformer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers_encoder,num_layers_decoder):
        super().__init__()
        self.encoder = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob,i)
                                     for i in range(num_layers_encoder)])
        
        self.decoder = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob,i) 
                                          for i in range(num_layers_decoder)])

    def forward(self, x,y,mask):
        print('-------------------------------ENCODER ACTIVATED----------------------------------------------------','\n\n')
        encoder_output = self.encoder(x)
        #print('encoder output : ',encoder_output)
        print('-------------------------------ENCODER COMPLETED-----------------------------------------------------','\n\n\n')
        print('-------------------------------DECODER ACTIVATED----------------------------------------------------','\n\n')
        decoder_output = self.decoder(encoder_output, y, mask)
        #print('decoder output : ',decoder_output)
        print('--------------------------------DECODER COMPLETED---------------------------------------------------','\n\n\n')
        return decoder_output
        
        

d_model = 128
num_heads = 8
drop_prob = 0.1
batch_size = 30
max_sequence_length = 200
ffn_hidden = 2048
num_layers_encoder = 5
num_layers_decoder = 5


x = torch.randn( (batch_size, max_sequence_length, d_model) ) # input positional encoded
y = torch.randn( (batch_size, max_sequence_length, d_model) ) # output sentence positional encoded 

mask = torch.full([max_sequence_length, max_sequence_length] , float('-inf'))
mask = torch.triu(mask, diagonal=1) # Mask initialization for masked attention

model = Transformer(d_model = 128,ffn_hidden = 2048,num_heads = 8,drop_prob = 0.1,num_layers_encoder = 5,num_layers_decoder = 5)
decoder_output = model(x,y,mask)
