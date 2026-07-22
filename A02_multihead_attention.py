import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, attn_mask):
        """
            Q: [batch_size, n_heads, len_q, d_model]
            K: [batch_size, n_heads, len_k, d_model]
            V: [batch_size, n_heads, len_v(=len_k), d_v]
            attn_mask: [batch_size, n_heads, seq_len, seq_len]
        """
        # 计算注意力分数, [batch_size, n_head, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_model)
        
        # 填充掩码矩阵
        scores.masked_fill_(attn_mask, -1e12)
        
        # 对最后一维求softmax，得到注意力权重
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        
        # 计算context向量, [batch_size, n_head, len_q, d_v]
        context = torch.matmul(attn, V)
        
        # 返回context向量和注意力权重矩阵(只用于调试)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, dropout=0.1, qkv_bias=False):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        # 线性层
        self.W_Q = nn.Linear(d_model, n_heads * d_k, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, n_heads * d_k, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, n_heads * d_v, bias=qkv_bias)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=qkv_bias)
        # 层归一化模块
        self.layernorm = nn.LayerNorm(d_model)
        # 缩放点积注意力模块
        self.attention = ScaledDotProductAttention(d_model, dropout)
    
    def forward(self, input_q, input_k, input_v, attn_mask):
        """
            input_q: [batch_size, len_q, d_model]
            input_k: [batch_size, len_k, d_model]
            input_v: [batch_size, len_v, d_model]
        """
        residual, batch_size = input_q, input_q.size(0)
        # 获取Q/K/V矩阵
        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_q).view(
            batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k]
        K = self.W_K(input_k).view(
            batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_v).view(
            batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        # 扩充掩码矩阵为四维
        # attn_mask: [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        
        # 计算注意力context向量
        # context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context, attn = self.attention(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)

        # 投影回d_model维输出
        outputs = self.fc(context)

        return self.layernorm(outputs + residual), attn


def test_multihead_attention():
    # ==================== 参数设置 ====================
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    d_k = d_v = d_model // n_heads  # 64

    # ==================== 设备设置 ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # ==================== 创建模型 ====================
    model = MultiHeadAttention(
        n_heads=n_heads,
        d_model=d_model,
        d_k=d_k,
        d_v=d_v,
        dropout=0.1
    ).to(device)
    
    # ==================== 创建输入数据 ====================
    # 自注意力通常 Q=K=V
    x = torch.randn(batch_size, seq_len, d_model, device=device)   # 输入
    
    # 创建注意力掩码（padding mask 示例）
    # 假设后3个token是padding
    pad_mask = torch.ones(batch_size, seq_len, seq_len, device=device)
    pad_mask[:, :, -3:] = 0          # 把最后3列设为padding
    pad_mask[:, -3:, :] = 0          # 把最后3行也设为padding
    attn_mask = (pad_mask == 0)      # True 表示要被mask掉的位置
    
    # ==================== 前向传播 ====================
    model.eval()   # 测试模式
    with torch.no_grad():
        output, attn = model(x, x, x, attn_mask)
    
    # ==================== 检查结果 ====================
    print("✅ 测试完成！")
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出是否与输入形状一致: {output.shape == x.shape}")
    
    # 检查数值是否正常（不是nan或inf）
    print(f"输出包含NaN: {torch.isnan(output).any()}")
    print(f"输出包含Inf: {torch.isinf(output).any()}")
    
    # 打印注意力权重信息
    print(f"注意力权重均值: {attn.mean().item():.6f}")
    print(f"注意力权重最大值: {attn.max().item():.6f}")
    print(f"注意力权重最小值: {attn.min().item():.6f}")
    print(f"单行注意力权重示例:\n{attn[0, 0, 0, :]}")   # 查看第一个head的第一行
    
    return output


if __name__ == "__main__":
    test_multihead_attention()