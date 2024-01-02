import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.heads_dim = embed_size // heads

        assert(self.heads_dim * heads == embed_size), "Embed size needs to be div by heads"

        self.values = nn.Linear(self.heads_dim, self.heads_dim, bias= False)
        self.keys = nn.Linear(self.heads_dim, self.heads_dim, bias= False)
        self.queries =  nn.Linear(self.heads_dim, self.heads_dim, bias= False)

        self.fc_out = nn.Linear(heads * self.heads_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.heads_dim)
        keys = keys.reshape(N, key_len, self.heads, self.heads_dim)
        queries = query.reshape(N, query_len, self.heads, self.heads_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum('nqhd,nkhd->nhqk', [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim = 3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.heads_dim)
        out = self.fc_out(out)

        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embeded_size, heads, dropout, forward_expansionn):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embeded_size, heads)

        self.norm1 = nn.LayerNorm(embeded_size)
        self.norm2 = nn.LayerNorm(embeded_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embeded_size, embeded_size*forward_expansionn),
            nn.ReLU(),
            nn.Linear(embeded_size*forward_expansionn, embeded_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))

        feed_forward = self.feed_forward(x)
        out = self.dropout(self.norm2(feed_forward + x))

        return out
    
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads,
                forward_expansion, dropout, max_length, device = None):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_len = x.shape
        if self.device:
            position = torch.arrange(0, seq_len).expand(N, seq_len).to(self.device)
        else:
            position = torch.arrange(0, seq_len).expand(N, seq_len)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(position))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansionn, dropout, device= None):
        super(DecoderBlock, self).__init__()

        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansionn
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(x + attention))
        out = self.transformer_block(value, key, query, src_mask)

        return out
    

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, 
                 forward_expansionn, dropout, max_length, device = None):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansionn, dropout, device)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_len = x.shape
        if self.device:
            position = nn.arrange(0, seq_len).expand(N, seq_len).to(self.device)
        else:
            position = nn.arrange(0, seq_len).expand(N, seq_len)

        x = self.dropout(self.word_embedding(x) + self.position_embedding(position))

        for layers in self.layers:
            x = layers(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx,
                 embed_size = 256, num_layers= 6, forward_expansionn= 4, heads= 8, 
                 dropout= 0, device= None, max_length= 100) -> None:
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, 
                               forward_expansionn, dropout, max_length, device)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads, 
                               forward_expansionn, dropout, max_length, device)
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        if self.device:
            return src_mask.to(self.device)
        else:
            return src_mask
        
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        if self.device:
            return trg_mask.to(self.device)
        else:
            return trg_mask
        
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)

        return out
