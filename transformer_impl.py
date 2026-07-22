import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,
                 context_length, dropout, num_heads, qkv_bias=False):
        super(MultiHeadAttention, self).__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        
        self.d_out = d_out  # 输出层维度
        self.num_heads = num_heads  # 注意力头数
        self.head_dim = d_out // num_heads  # 每个头的维度

        # 定义线性变换层
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)
        # 注册一个上三角全为1的掩码矩阵
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # (b, num_tokens, d_in) -> (b, num_tokens, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # 通过添加一个 num_heads维度来隐式地分隔矩阵。然后展开最后一个维度：(b, num_tokens, d_out) -> (b,  num_tokens,  num_heads,  head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        
        # 从形状(b, num_tokens, num_heads, head_dim)转换到(b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # 计算注意力权重矩阵，使用掩码来填充注意力分数 
        attn_scores = torch.matmul(queries, keys.transpose(2, 3))
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]    # 被截断为token数量的掩码 
        attn_scores.masked_fill_(mask_bool, -torch.inf)     # 因果注意力掩码
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # (b, num_tokens,  n_heads, head_dim)
        context_vec = torch.matmul(attn_weights, values).transpose(1, 2)
        
        # 组合头，其中self.d_out = self.num_heads *  self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        
        # 添加一个可选的线性投影
        context_vec = self.out_proj(context_vec)
        
        return context_vec


class PoswiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, bias=False):
        super(PoswiseFeedForward, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=bias)
        )
        self.layernorm = nn.LayerNorm(d_model)
    
    def forward(self, inputs):
        residual = inputs
        outputs = self.fc(inputs)
        return self.layernorm(residual + outputs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)    # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # [max_len, d_model] -> [max_len, 1, d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
            x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
