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

        energy = torch.einsum('nqhd,nkhd->nhqk', [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim = 3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.heads_dim)
        out = self.fc_out(out)

        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embeded_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embeded_size, heads)

        self.norm1 = nn.LayerNorm(embeded_size)
        self.norm2 = nn.LayerNorm(embeded_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embeded_size, embeded_size*forward_expansion),
            nn.ReLU(),
            nn.Linear(embeded_size*forward_expansion, embeded_size)
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
                forward_expansio, dropout, max_lenght, device = None):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_lenght, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansio)
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
    def __init__(self, embed_size, heads, forward_expansion, dropout, device= None):
        super(DecoderBlock, self).__init__()

        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(x + attention))
        out = self.transformer_block(value, key, query, src_mask)

        return out
    

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, 
                 forward_expansion, dropout, max_lenght, device = None):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_lenght, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
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


class Transformer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)



