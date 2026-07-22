import torch
import torch.nn as nn
from A02_multihead_attention import MultiHeadAttention
from A03_positional_encoding import PositionalEncoding
from A04_poswise_feed_forward import PoswiseFeedForward
from A05_attention_mask import get_attn_pad_mask


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, d_ff, 
                 dropout=0, qkv_bias=False, ff_bias=False):
        super(EncoderLayer, self).__init__()
        # 自注意力模块
        self.self_attention = MultiHeadAttention(
            n_heads, d_model, d_k, d_v, dropout, qkv_bias)
        # MLP模块
        self.ffn = PoswiseFeedForward(d_model, d_ff, ff_bias)
    
    def forward(self, enc_inputs, enc_self_attn_mask):
        """
            enc_inputs: [batch_size, src_len, d_model]
            enc_self_attn_mask: [batch_size, src_len, src_len]  mask矩阵(pad mask or sequence mask)
            enc_outputs: [batch_size, src_len, d_model]
        """
        enc_outputs, attn = self.self_attention(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.ffn(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, n_layers, src_vocab_size, n_heads, d_model, d_k, d_v, d_ff, 
                 dropout=0, qkv_bias=False, ff_bias=False, padding_idx=0):
        super(Encoder, self).__init__()
        self.padding_idx = padding_idx
        self.tokenizer = nn.Embedding(src_vocab_size, d_model)    # 转化为词token
        self.pos_encoding = PositionalEncoding(d_model, dropout)     # 位置编码
        # 堆叠n_layers层
        self.layers = nn.ModuleList([
            EncoderLayer(n_heads, d_model, d_k, d_v, d_ff, dropout, qkv_bias, ff_bias)
             for _ in range(n_layers)
        ])
    
    def forward(self, enc_inputs):
        """
            enc_inputs: [batch_size, src_len]
            enc_outputs: [batch_size, src_len, d_model], enc_self_attns
        """
        # [batch_size, src_len, d_model]
        # 词嵌入，得到词token
        enc_outputs = self.tokenizer(enc_inputs)
        # [batch_size, src_len, d_model] -> [src_len, batch_size, d_model] -> [batch_size, src_len, d_model]
        # 位置编码
        enc_outputs = self.pos_encoding(enc_outputs.transpose(0, 1)).transpose(0, 1)
        # 计算掩码矩阵: [batch_size, src_len, src_len]
        enc_attn_pad_mask = get_attn_pad_mask(enc_inputs, enc_inputs, self.padding_idx)
        # 注意力权重列表，存放中间过程用于可视化，生成像论文里面的注意力权重热力图
        enc_self_attns = []
        for layer in self.layers:
            # 过编码层
            enc_outputs, attn = layer(enc_outputs, enc_attn_pad_mask)
            # 保存中间过程用于可视化
            enc_self_attns.append(attn)
        return enc_outputs, enc_self_attns