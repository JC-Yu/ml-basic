import torch
import torch.utils.data as Data
from A10_transformer import Transformer


class WordDataSet(Data.Dataset):
    """自定义DataLoader"""
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(WordDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


class TransformerLauncher:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 100
    batch_size = 2
    # 训练集
    sentences = [
        # 中文和英语的单词个数不要求相同
        # enc_input                dec_input           dec_output
        ['我 有 一 个 好 朋 友 P', 'S I have a good friend .', 'I have a good friend . E'],
        ['我 有 零 个 女 朋 友 P', 'S I have zero girl friend .', 'I have zero girl friend . E'],
        ['我 有 一 个 男 朋 友 P', 'S I have a boy friend .', 'I have a boy friend . E']
    ]
    padding_index = 0
    # 中文和英语的单词要分开建立词库
    # Padding Should be Zero
    src_vocab = {'P': 0, '我': 1, '有': 2, '一': 3,
                '个': 4, '好': 5, '朋': 6, '友': 7, '零': 8, '女': 9, '男': 10}
    src_idx2word = {i: w for i, w in enumerate(src_vocab)}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'I': 1, 'have': 2, 'a': 3, 'good': 4,
                'friend': 5, 'zero': 6, 'girl': 7,  'boy': 8, 'S': 9, 'E': 10, '.': 11}
    idx2word = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)

    # 测试集
    test_sentences = [
            # enc_input                dec_input           dec_output
            ['我 有 零 个 女 朋 友 P', '', '']
        ]

    src_len = 8  # （源句子的长度）enc_input max sequence length
    tgt_len = 7  # dec_input(=dec_output) max sequence length

    # Transformer Parameters
    d_model = 512  # Embedding Size（token embedding和position编码的维度）
    # FeedForward dimension (两次线性层中的隐藏层 512->2048->512，线性层是用来做特征提取的），当然最后会再接一个projection层
    d_ff = 2048
    d_k = d_v = 64  # dimension of K(=Q), V（Q和K的维度需要相同，这里为了方便让K=V）
    n_layers = 6  # number of Encoder of Decoder Layer（Block的个数）
    n_heads = 8  # number of heads in Multi-Head Attention（有几套头）

    def make_train_data(self):
        """把单词序列转换为数字序列"""
        enc_inputs, dec_inputs, dec_outputs = [], [], []
        for i in range(len(self.sentences)):
    
            enc_input = [[self.src_vocab[n] for n in self.sentences[i][0].split()]]
            dec_input = [[self.tgt_vocab[n] for n in self.sentences[i][1].split()]]
            dec_output = [[self.tgt_vocab[n] for n in self.sentences[i][2].split()]]

            #[[1, 2, 3, 4, 5, 6, 7, 0], [1, 2, 8, 4, 9, 6, 7, 0], [1, 2, 3, 4, 10, 6, 7, 0]]
            enc_inputs.extend(enc_input)
            #[[9, 1, 2, 3, 4, 5, 11], [9, 1, 2, 6, 7, 5, 11], [9, 1, 2, 3, 8, 5, 11]]
            dec_inputs.extend(dec_input)
            #[[1, 2, 3, 4, 5, 11, 10], [1, 2, 6, 7, 5, 11, 10], [1, 2, 3, 8, 5, 11, 10]]
            dec_outputs.extend(dec_output)

        return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)
    
    def make_eval_data(self):
        """把单词序列转换为数字序列"""
        enc_inputs, dec_inputs, dec_outputs = [], [], []
        for i in range(len(self.test_sentences)):
    
            enc_input = [[self.src_vocab[n] for n in self.test_sentences[i][0].split()]]
            dec_input = [[self.tgt_vocab[n] for n in self.test_sentences[i][1].split()]]
            dec_output = [[self.tgt_vocab[n] for n in self.test_sentences[i][2].split()]]

            #[[1, 2, 3, 4, 5, 6, 7, 0], [1, 2, 8, 4, 9, 6, 7, 0], [1, 2, 3, 4, 10, 6, 7, 0]]
            enc_inputs.extend(enc_input)
            #[[9, 1, 2, 3, 4, 5, 11], [9, 1, 2, 6, 7, 5, 11], [9, 1, 2, 3, 8, 5, 11]]
            dec_inputs.extend(dec_input)
            #[[1, 2, 3, 4, 5, 11, 10], [1, 2, 6, 7, 5, 11, 10], [1, 2, 3, 8, 5, 11, 10]]
            dec_outputs.extend(dec_output)

        return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)
    
    def make_dataloader(self, enc_inputs, dec_inputs, dec_outputs, shuffle=True):
        return Data.DataLoader(
            WordDataSet(enc_inputs, dec_inputs, dec_outputs), self.batch_size, shuffle=shuffle)
    
    def make_transformer(self):
        return Transformer(
            n_layers=self.n_layers,
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            n_heads=self.n_heads,
            d_model=self.d_model,
            d_k=self.d_k,
            d_v=self.d_v,
            d_ff=self.d_ff,
            dropout=0,
            padding_idx=self.padding_index
        )