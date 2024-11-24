import torch
from torch import nn


class Model(nn.Module):
    def __init__(self,num_features=20):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(batch_first=True, input_size=num_features, hidden_size=50, num_layers=1)
        self.attenion = Attention3dBlock()
        self.linear = nn.Sequential(
            nn.Linear(in_features=3000, out_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=1)
        )
        self.multihead_attention = MultiHeadAttention(embed_dim=50, num_heads=1)

    def forward(self, inputs):
        x, (hn, cn) = self.lstm(inputs)
        # print(x.shape)
        y = self.multihead_attention(x)
        x = self.attenion(x)
        # flatten
        y = y.reshape(-1,1500)
        x = x.reshape(-1, 1500)
        x = torch.concat((x,y),dim=1)
        x = self.linear(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear layers to project the input to query, key, and value spaces for each head
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # Linear layer to combine the outputs from all heads
        self.out = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        
        # Linear projections
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Split each projection into multiple heads and reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention for each head
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = self.softmax(scores)
        attn_output = torch.matmul(attn_weights, v)
        
        # Concatenate all heads and apply the final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out(attn_output)
        
        return output
    
class Attention3dBlock(nn.Module):
    def __init__(self):
        super(Attention3dBlock, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(in_features=30, out_features=30),
            nn.Softmax(dim=2),
        )

    # inputs: batch size * window size(time step) * lstm output dims
    def forward(self, inputs):
        x = inputs.permute(0, 2, 1)
        x = self.linear(x)
        x_probs = x.permute(0, 2, 1)
        # print(torch.sum(x_probs.item()))
        output = x_probs * inputs
        return output
