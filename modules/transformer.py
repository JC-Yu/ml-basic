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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 位置编码矩阵
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)    # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # [max_len, d_model] -> [max_len, 1, d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # 注册为缓冲区矩阵
    
    def forward(self, x):
        """
            x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PoswiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, bias=False):
        super(PoswiseFeedForward, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias)
        )
        self.layernorm = nn.LayerNorm(d_model)
    
    def forward(self, inputs):
        """
            inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        outputs = self.fc(inputs)
        return self.layernorm(residual + outputs)


class MaskGenerator:
    def __init__(self):
        pass

    @staticmethod
    def get_attn_pad_mask(seq_q, seq_k, padding_idx: int = 0):
        # pad mask的作用：在对value向量加权平均的时候，可以让pad对应的alpha_ij=0，这样注意力就不会考虑到pad向量
        """这里的q,k表示的是两个序列（跟注意力机制的q,k没有关系），例如encoder_inputs (x1,x2,..xm)和encoder_inputs (x1,x2..xm)
        encoder和decoder都可能调用这个函数，所以seq_len视情况而定
            seq_q: [batch_size, len_q]
            seq_k: [batch_size, len_k]
            pad_mask: [batch_size, len_q, len_k]
        """
        batch_size, len_q = seq_q.shape
        batch_size, len_k = seq_k.shape
        # [batch_size, len_k]
        pad_mask = (seq_k == padding_idx)
        # [batch_size, len_k] -> [batch_size, len_q, len_k]
        pad_mask = pad_mask.unsqueeze(1).expand(batch_size, len_q, len_k)
        return pad_mask

    @staticmethod
    def get_attn_subsequence_mask(seq):
        """
            生成子序列掩码(因果掩码)
            seq: [batch_size, seq_len]
            subsequence_mask: [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len = seq.shape
        subsequence_mask = torch.triu(
            torch.ones([batch_size, seq_len, seq_len], dtype=bool, device=seq.device),
            diagonal=1
        )
        return subsequence_mask


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
        enc_attn_pad_mask = MaskGenerator.get_attn_pad_mask(
            enc_inputs, enc_inputs, self.padding_idx)
        # 注意力权重列表，存放中间过程用于可视化，生成像论文里面的注意力权重热力图
        enc_self_attns = []
        for layer in self.layers:
            # 过编码层
            enc_outputs, attn = layer(enc_outputs, enc_attn_pad_mask)
            # 保存中间过程用于可视化
            enc_self_attns.append(attn)
        return enc_outputs, enc_self_attns


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
        dec_self_attn_pad_mask = MaskGenerator.get_attn_pad_mask(dec_inputs, dec_inputs, self.padding_idx)
        # Masked Self_Attention：当前时刻是看不到未来的信息的 [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = MaskGenerator.get_attn_subsequence_mask(dec_inputs)
        # 将两个掩码作逻辑或，合并起来作为自注意力掩码矩阵
        dec_self_attn_mask = dec_self_attn_pad_mask | dec_self_attn_subsequence_mask

        # 获取交叉注意力的掩码矩阵
        dec_enc_attn_mask = MaskGenerator.get_attn_pad_mask(dec_inputs, enc_inputs, self.padding_idx)

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