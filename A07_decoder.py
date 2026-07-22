import torch
import torch.nn as nn
from A02_multihead_attention import MultiHeadAttention
from A03_positional_encoding import PositionalEncoding
from A04_poswise_feed_forward import PoswiseFeedForward
from A05_attention_mask import get_attn_pad_mask, get_attn_subsequence_mask


class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, d_ff, 
                 dropout=0, qkv_bias=False, ff_bias=False):
        super(DecoderLayer, self).__init__()
        # decoder到decoder的自注意力模块
        self.dec_self_attn = MultiHeadAttention(
            n_heads, d_model, d_k, d_v, dropout, qkv_bias
        )
        # encoder到decoder的交叉注意力模块
        self.dec_enc_attn = MultiHeadAttention(
            n_heads, d_model, d_k, d_v, dropout, qkv_bias
        )
        self.ffn = PoswiseFeedForward(d_model, d_ff, ff_bias)
    
    def forward(self, enc_outputs, dec_inputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
            enc_outputs: [batch_size, src_len, d_model]
            dec_inputs: [batch_size, tgt_len, d_model]
            dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
            dec_enc_attn_mask: [batch_size, tgt_len, src_len]
            dec_outputs: [batch_size, tgt_len, d_model]
        """
        # 先过自注意力层，QKV都来自解码器自己的输入dec_inputs
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(
            dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask
        )
        # 再过交叉注意力层，Q来自dec_outputs，KV是enc_outputs
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(
            dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask
        )
        # 最后过全连接层
        # [batch_size, tgt_len, d_model]
        dec_outputs = self.ffn(dec_outputs)

        # dec_self_attn, dec_enc_attn这两个是为了可视化的
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, n_layers, tgt_vocab_size, n_heads, d_model, d_k, d_v, d_ff, 
                 dropout=0, qkv_bias=False, ff_bias=False, padding_idx=0):
        super(Decoder, self).__init__()
        self.padding_idx = padding_idx
        self.tokenizer = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(n_heads, d_model, d_k, d_v, d_ff, dropout, qkv_bias, ff_bias)
             for _ in range(n_layers)]
        )
    
    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
            dec_inputs: [batch_size, tgt_len]
            enc_inputs: [batch_size, src_len]
            enc_outputs: [batch_size, src_len, d_model]
        """
        # 获取解码器输入的词嵌入
        # [batch_size, tgt_len, d_model]
        dec_outputs = self.tokenizer(dec_inputs)
        
        # 做位置编码
        # [batch_size, tgt_len, d_model] -> [tgt_len, batch_size, d_model] -> [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_encoding(dec_outputs.transpose(0, 1)).transpose(0, 1)
        
        # 获取自注意力的掩码矩阵
        # Decoder输入序列的pad mask矩阵 [batch_size, tgt_len, tgt_len]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.padding_idx)
        # Masked Self_Attention：当前时刻是看不到未来的信息的 [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)
        # 将两个掩码作逻辑或，合并起来作为自注意力掩码矩阵
        dec_self_attn_mask = dec_self_attn_pad_mask | dec_self_attn_subsequence_mask

        # 获取交叉注意力的掩码矩阵
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs, self.padding_idx)

        # 注意力权重列表，用于可视化热力图
        dec_self_attns = []
        dec_enc_attns = []
        for layer in self.layers:
            # 逐层过解码层
            dec_outputs, dec_self_attn, dec_enc_attn = layer(
                enc_outputs, dec_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            # 保存中间结果用于可视化
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        
        return dec_outputs, dec_self_attns, dec_enc_attns