import torch
import torch.nn as nn


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


def test_masks():
    # ==================== 测试数据 ====================
    batch_size = 2
    src_len = 6
    tgt_len = 5
    
    # 模拟输入序列，0 表示 padding token
    src_seq = torch.tensor([
        [1, 2, 3, 4, 0, 0],      # 有效长度4
        [5, 6, 7, 0, 0, 0]       # 有效长度3
    ])  # shape: [2, 6]
    
    tgt_seq = torch.tensor([
        [1, 2, 3, 0, 0],          # 有效长度3
        [4, 5, 0, 0, 0]           # 有效长度2
    ])  # shape: [2, 5]
    
    print("=== 测试数据 ===")
    print(f"src_seq:\n{src_seq}")
    print(f"tgt_seq:\n{tgt_seq}\n")
    
    # ==================== 测试 Padding Mask ====================
    print("=== Padding Mask (get_attn_pad_mask) ===")
    
    # 自注意力时通常 src_q 和 src_k 相同
    pad_mask_src = get_attn_pad_mask(src_seq, src_seq, padding_idx=0)
    print(f"src self-attention pad_mask shape: {pad_mask_src.shape}")
    print("src self-attention pad_mask:")
    print(pad_mask_src[0])   # 打印第一个 batch 的 mask
    
    # Encoder-Decoder Cross Attention
    pad_mask_cross = get_attn_pad_mask(tgt_seq, src_seq, padding_idx=0)
    print(f"\ncross-attention pad_mask shape: {pad_mask_cross.shape}")
    print("cross-attention pad_mask (batch 0):")
    print(pad_mask_cross[0])
    
    # ==================== 测试 Subsequence Mask ====================
    print("\n=== Subsequence Mask (因果掩码) ===")
    sub_mask = get_attn_subsequence_mask(tgt_seq)
    print(f"subsequence_mask shape: {sub_mask.shape}")
    print("subsequence_mask (batch 0):")
    print(sub_mask[0].int())   # 转成 0/1 显示更清晰
    
    # ==================== 组合掩码（Decoder Self-Attention 常用）===
    print("\n=== Decoder Self-Attention 组合掩码示例 ===")
    # 通常 decoder self-attn 需要 pad_mask + subsequence_mask
    pad_mask_tgt = get_attn_pad_mask(tgt_seq, tgt_seq, padding_idx=0)
    combined_mask = pad_mask_tgt | sub_mask      # 逻辑或
    print("combined_mask (batch 0):")
    print(combined_mask[0].int())


if __name__ == "__main__":
    test_masks()
