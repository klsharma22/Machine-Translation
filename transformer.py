import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizer, BertModel,AutoTokenizer, AutoModel

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
    attention = attention.to(device='cpu')
    torch.cuda.empty_cache()

     # 30 x 8 x 200 x 200
    values = torch.matmul(attention.to(device='cuda:0'), v.to(device='cuda:0')).to(device='cuda:0') # 30 x 8 x 200 x 64
    print('values in ', values.device)
    attention = attention.to(device='cpu')
    values=values.to('cpu')
    torch.cuda.empty_cache()
    del attention

    return values

class PositionalEncoding(nn.Module):
    def __init__(self,batch_size):
        super().__init__()
        self.batch_size = batch_size
    
    def positional_encoding(self,embeddings):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        records,max_sequence_length,d_model = embeddings.size()

        even_i = torch.arange(0 , d_model , 2).float()
        even_denominator = torch.pow(10000, even_i/d_model)
        odd_i = torch.arange(1 , d_model , 2).float()
        odd_denominator = torch.pow(10000, (odd_i -1)/d_model)

        positions = torch.arange(max_sequence_length,dtype=torch.float).reshape(max_sequence_length,1)

        even_pe = torch.sin(positions/even_denominator)
        odd_pe = torch.sin(positions/even_denominator)
        stacked = torch.stack([even_pe , odd_pe] , dim  = 2)
        PE = torch.flatten(stacked,start_dim=1,end_dim=2)
        PE = torch.tile(PE,(self.batch_size,1,1))
        test_list=[]

        for i in tqdm(range(0 ,records,self.batch_size), "Positional_Encoding", colour= "green"):
            batch = embeddings[i:i+self.batch_size]
            test_list.append(batch + PE)
        test_list = torch.stack(test_list).to(device=device)
        test_list = test_list.to(device='cpu')
        torch.cuda.empty_cache()
        test_list= torch.flatten(test_list,start_dim=0,end_dim=1).to(device=device)
        test_list = test_list.to(device='cpu')
        torch.cuda.empty_cache()
        print('POSITIONAL ENCODING IS DONE','\n',test_list.size())
        return test_list
    
class Embedding(nn.Module):
    def __init__(self,max_seq_length,d_model,batch_size):
        super().__init__()
        self.max_len = max_seq_length
        self.d_model = d_model
        self.batch_size = batch_size
        self.embedding_model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")


    def text_embedding(self,batch_tokens):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.embedding_model = self.embedding_model.to(device=device)

        # Reduce the batch size if the input is too large for GPU memory
        batch_size = len(batch_tokens)
        max_batch_size = 50  # You can adjust this value based on your GPU memory capacity

        while batch_size > max_batch_size:
            batch_tokens = batch_tokens[:batch_size // 2]  # Halve the batch size
            batch_size = len(batch_tokens)

        batch_padded_tokens = [tokens + [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
                            for tokens in batch_tokens]

        tokens_tensor = torch.tensor(batch_padded_tokens).to(device=device)
        with torch.no_grad():
            output = self.embedding_model(tokens_tensor)
            embeddings = output.last_hidden_state
            embeddings=embeddings.to('cpu')
            torch.cuda.empty_cache()
            self.embedding_model

        return embeddings
    

    def get_embeddings(self,tokens):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        embedding_trans = []
        for i in tqdm(range(0, len(tokens), self.batch_size), "Embedding", colour= "green"):
            batch_token = tokens[i : i+self.batch_size]
            embedding_trans.extend(self.text_embedding(batch_token))
        embedding_trans = torch.stack(embedding_trans).to(device=device)
        embedding_trans = embedding_trans.to(device='cpu')
        torch.cuda.empty_cache()
        return embedding_trans

class SentenceEmbedding(nn.Module):
    "For a given sentence, create an embedding"
    def __init__(self, max_sequence_length, d_model,batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.embedding = Embedding(self.max_sequence_length, self.d_model,self.batch_size)
        self.position_encoder = PositionalEncoding(self.batch_size)
    

    def tokenize(self,sentence):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tokens = [self.tokenizer.encode(text,add_special_tokens = True,padding='max_length',max_length=self.max_sequence_length) for text in sentence]
        return tokens
    
    def forward(self, x): # sentence
        x = self.tokenize(x)
        print('tokenization done','\n')
        x = self.embedding.get_embeddings(x)
        x = self.position_encoder.positional_encoding(x)
        print('SENTENCE EMBEDDING DONE','\n',x.size())
        return x
    

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

        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim) # 30 x 200 x 8 x 192
        qkv = qkv.to(device='cpu')
        torch.cuda.empty_cache() 
        print('qkv in',qkv.device)
        #print(f"qkv after reshape .size(): {qkv.size()}")

        qkv = qkv.permute(0, 2, 1, 3) # 30 x 8 x 200 x 192
        qkv = qkv.to(device='cpu')
        torch.cuda.empty_cache() 
        print('qkv in',qkv.device)
        #print(f"qkv after permutation: {qkv.size()}")

        q, k, v = qkv.chunk(3, dim=-1) # q: 30 x 8 x 200 x 64, k: 30 x 8 x 200 x 64, v: 30 x 8 x 200 x 64
        
        #print(f"q: {q.size()}, k:{k.size()}, v:{v.size()}")

        values = scaled_dot_product(q, k, v, mask) # values: 30 x 8 x 200 x 64
        #print(f"values: {values.size()}, attention:{attention.size()}")
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim) # 30 x 200 x 512
        #print(f"values after reshaping: {values.size()}")
        out = self.linear_layer(values.to(device='cuda:0')) # 30 x 200 x 512
        out = out.to(device='cpu')
        torch.cuda.empty_cache()
        print(f"Multi headed hattention done , size=: {out.size()}",'\n')
        return out # 30 x 200 x 512
    


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape):
        super().__init__()
        self.parameters_shape=parameters_shape
        self.norm = nn.LayerNorm(self.parameters_shape).to(device='cuda:0')

    def forward(self, inputs):
        inputs = inputs.to(device='cuda:0')
        out = self.norm(inputs)
        out = out.to(device='cpu')
        torch.cuda.empty_cache()
        print(f"Layer normalization done , Size=: {out.size()}",'\n')
        return out


  
class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden).to(device='cuda:0')
        self.linear2 = nn.Linear(hidden, d_model).to(device='cuda:0')
        self.relu = nn.ReLU().to(device='cuda:0')
        self.dropout = nn.Dropout(p=drop_prob).to(device='cuda:0')

    def forward(self, x):
        x = self.linear1(x.to(device='cuda'))
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

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob,i=0):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob).to(get_device())
        self.i_th_encoder = i+1

    def forward(self, x, self_attention_mask):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(f'------ ENCODER LAYER NUMBER {self.i_th_encoder}----------')
        residual_x = x.clone()
        print("ATTENTION 1",)
        x = self.attention.forward(x, mask=None)
        print(x.device)
        torch.cuda.empty_cache()

        print("DROPOUT 1",'\n')
        x = self.dropout1(x.to(device=device))
        x=x.to(device='cpu')
        torch.cuda.empty_cache()

        print("-ADD AND LAYER NORMALIZATION 1 -")
        x = self.norm1(x + residual_x)
        x=x.to(device='cpu')
        torch.cuda.empty_cache()

        residual_x = x
        print(" FEED FORWARD NETWORK")
        x = self.ffn(x)
        x=x.to(device='cpu')
        torch.cuda.empty_cache()

        print("-DROPOUT 2 -",'\n')
        x = self.dropout2(x).to(device=device)
        x=x.to(device='cpu')
        torch.cuda.empty_cache()

        print("-ADD AND LAYER NORMALIZATION 2-")
        x = self.norm2(x + residual_x)
        x=x.to(device='cpu')
        torch.cuda.empty_cache()
        return x
    
class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask  = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x

class Encoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 batch_size
                ):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model,batch_size)
        self.encoder = EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
        self.num_layers = num_layers
                                     

    def forward(self, x, self_attention_mask):
        x = self.sentence_embedding.forward(x)
        print('MEMORY USAGE AFTER ENCODING','\n',torch.cuda.memory_summary())
        for i in range(self.num_layers):
            x = self.encoder(x, self_attention_mask)
            x = x.to(device = 'cpu')
            torch.cuda.empty_cache()
        return x


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
        values = scaled_dot_product(q, k, v, mask) #  30 x 8 x 200 x 64
        #print(f"values: {values.size()}, attention:{attention.size()}")
        values = values.reshape(batch_size, sequence_length, d_model).to(device='cuda:0') #  30 x 200 x 512
        values = values.to('cpu')
        torch.cuda.empty_cache()
        out = self.linear_layer(values.to(device='cuda:0'))
        out = out.to(device='cpu')  #  30 x 200 x 512
        torch.cuda.empty_cache()
        print(f"Cross attention completed size=: {out.size()}",'\n')
        return out  #  30 x 200 x 512


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob,i):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm1 = LayerNormalization(parameters_shape=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob).to(device='cuda:0')

        self.cross_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm2 = LayerNormalization(parameters_shape=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob).to(device='cuda:0')

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.layer_norm3 = LayerNormalization(parameters_shape=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob).to(device='cuda:0')
        self.i_th_decoder = i+1

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        print(f'------- DEOCDER LAYER NUMBER {self.i_th_decoder}-----------','\n')
        _y = y # 30 x 200 x 512
        print("MASKED SELF ATTENTION")
        y = self.self_attention(y, mask=self_attention_mask) # 30 x 200 x 512

        print("DROP OUT 1")
        y = self.dropout1(y.to(device='cuda:0')) # 30 x 200 x 512
        y = y.to(device='cpu')
        torch.cuda.empty_cache()

        print("ADD + LAYER NORMALIZATION 1")
        y = self.layer_norm1(y + _y) # 30 x 200 x 512
        _y = y # 30 x 200 x 512

        print("CROSS ATTENTION")
        y = self.cross_attention(x, y, mask=cross_attention_mask) #30 x 200 x 512

        print("DROP OUT 2",'\n')  #30 x 200 x 512
        y = self.dropout2(y.to(device='cuda:0'))
        y = y.to(device='cpu')
        torch.cuda.empty_cache()

        print("ADD + LAYER NORMALIZATION 2")
        y = self.layer_norm2(y + _y)  #30 x 200 x 512
        _y = y  #30 x 200 x 512

        print("FEED FORWARD 1")
        y = self.ffn(y) #30 x 200 x 512

        print("DROP OUT 3",'\n')
        y = self.dropout3(y.to(device='cuda:0')) #30 x 200 x 512
        y = y.to(device='cpu')
        torch.cuda.empty_cache()

        print("ADD + LAYER NORMALIZATION 3")
        y = self.layer_norm3(y + _y) #30 x 200 x 512
        return y #30 x 200 x 512


class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask)
        return y

class Decoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 batch_size):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model,batch_size)
        self.decoder = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob,i) for i in range(num_layers)])

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        y = self.sentence_embedding.forward(y)
        y = self.decoder(x, y, self_attention_mask, cross_attention_mask)
        return y


class Transformer(nn.Module):
    def __init__(self, 
                d_model, 
                ffn_hidden, 
                num_heads, 
                drop_prob, 
                max_sequence_length,
                num_layers_encoder,
                num_layers_decoder,
                batch_size
                ):
        super().__init__()
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers_encoder, max_sequence_length,batch_size)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers_decoder, max_sequence_length,batch_size)
        self.linear = nn.Linear(d_model,max_sequence_length)

    def forward(self, 
                x, 
                y, 
                encoder_self_attention_mask=None, 
                decoder_self_attention_mask=None, 
                decoder_cross_attention_mask=None
                ): # x, y are batch of sentences
        print('ENCODER ACTIVATED')
        x = self.encoder(x, encoder_self_attention_mask)
        print('ENCODER COMPLETED')
        print('MEMORY AFTER ENCODER ACTION','\n',(torch.cuda.memory_summary(abbreviated=True)))
        print('DECODER ACTIVATED')
        out = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask)
        out = self.linear(out)
        print('DECODER COMPLETED')
        print('MEMORY AFTER DECODER ACTION','\n',(torch.cuda.memory_summary(abbreviated=True)))
        return out


#x = torch.randn( (size, max_sequence_length, d_model) ) # input positional encoded
#y = torch.randn( (size, max_sequence_length, d_model) ) # output sentence positional encoded 
    


d_model = 128
num_heads = 8
drop_prob = 0.1
batch_size = 30
max_sequence_length = 104
ffn_hidden = 2048
num_layers_encoder = 1
num_layers_decoder = 1
transformer = Transformer(d_model = 128,
                    ffn_hidden = 256,
                    num_heads = 8,
                    drop_prob = 0.1,
                    max_sequence_length = 104,
                    num_layers_encoder = 2,
                    num_layers_decoder = 1,
                    batch_size=1
                    )

x = ["I'm really in a bind","I'm really surprised"]
y = ['Je suis vraiment dans le pétrin','Je suis très étonné.']
mask = torch.full([max_sequence_length, max_sequence_length] , float('-inf'))
mask = torch.triu(mask, diagonal=1) # Mask initialization for masked attention
out = transformer(x,y,mask)

