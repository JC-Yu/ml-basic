import torch
import torch.nn as nn
from A06_encoder import Encoder
from A07_decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, 
                 n_layers, 
                 src_vocab_size, 
                 tgt_vocab_size, 
                 n_heads, 
                 d_model, 
                 d_k, 
                 d_v, 
                 d_ff, 
                 dropout=0, 
                 qkv_bias=False, 
                 ff_bias=False,
                 proj_bias=False,
                 padding_idx=0):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            n_layers, src_vocab_size, n_heads, d_model, d_k, d_v, d_ff, 
            dropout, qkv_bias, ff_bias, padding_idx
        )
        self.decoder = Decoder(
            n_layers, tgt_vocab_size, n_heads, d_model, d_k, d_v, d_ff, 
            dropout, qkv_bias, ff_bias, padding_idx
        )
        # 解码器输出经线性层投影回tgt_vocab_size维再输出
        self.proj = nn.Linear(d_model, tgt_vocab_size, bias=proj_bias)

    def forward(self, enc_inputs, dec_inputs):
        """
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]
            dec_logits: [batch_size * tgt_len, tgt_vocab_size]
            enc_self_attns: [batch_size, src_len, d_model]
            dec_self_attns: [batch_size, tgt_len, d_model]
            dec_enc_attns: [batch_size, tgt_len, d_model]
        """
        # [batch_size, src_len, d_model]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # [batch_size, tgt_len, d_model]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        # [batch_size, tgt_len, d_model] -> [batch_size, tgt_len, tgt_vocab_size]
        dec_logits = self.proj(dec_outputs)
        # [batch_size, tgt_len, tgt_vocab_size] -> [batch_size * tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
