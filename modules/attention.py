import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
       
    def forward(self, x):
        """
            x: [batch_size, seq_len, d_in]
        """
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim = -1
        )
        context_vec = attn_weights @ values
        return context_vec


class CausualAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(             
            'mask',             
            torch.triu(torch.ones(context_length, context_length),             
            diagonal=1)
        )
       
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim = -1
        )
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,
                 context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
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
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
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
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # (b, num_tokens,  n_heads, head_dim)
        context_vec = torch.matmul(attn_weights, values).transpose(1, 2)
        
        # 组合头，其中self.d_out = self.num_heads *  self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        
        # 添加一个可选的线性投影
        context_vec = self.out_proj(context_vec)
        return context_vec