import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from modules.attention import SelfAttention, CausualAttention, MultiHeadAttention


def test_attention():
    # 创建一个维度为 [batch_size, seq_len, embed_dim] 的输入张量
    batch_size = 2
    seq_len = 4
    embed_dim = 8
    x = torch.rand(batch_size, seq_len, embed_dim, dtype=torch.float32)
    assert x.shape == (batch_size, seq_len, embed_dim), \
        f"Expected input shape {(batch_size, seq_len, embed_dim)}, but got {x.shape}"

    # 创建一个 Self Attention 实例
    self_attention = SelfAttention(d_in=embed_dim, d_out=embed_dim)
    # 前向传播
    output_self_attention = self_attention(x)
    # 检查输出形状是否正确
    assert output_self_attention.shape == (batch_size, seq_len, embed_dim), \
        f"Expected output shape {(batch_size, seq_len, embed_dim)}, but got {output_self_attention.shape}"

    # 创建一个 Causual Attention 实例
    causual_attention = CausualAttention(
        d_in=embed_dim, 
        d_out=embed_dim,
        context_length=seq_len, 
        dropout=0.1)
    # 前向传播
    output_causual_attention = causual_attention(x)
    # 检查输出形状是否正确
    assert output_causual_attention.shape == (batch_size, seq_len, embed_dim), \
        f"Expected output shape {(batch_size, seq_len, embed_dim)}, but got {output_causual_attention.shape}"

    # 创建一个 Multi-Head Attention 实例
    num_heads = 2
    multi_head_attention = MultiHeadAttention(
        d_in=embed_dim,
        d_out=embed_dim,
        context_length=seq_len,
        dropout=0.1,
        num_heads=num_heads)
    # 前向传播
    output_multi_head_attention = multi_head_attention(x)
    # 检查输出形状是否正确
    assert output_multi_head_attention.shape == (batch_size, seq_len, embed_dim), \
        f"Expected output shape {(batch_size, seq_len, embed_dim)}, but got {output_multi_head_attention.shape}"


if __name__ == "__main__":
    test_attention()
    print("All tests passed!")

    