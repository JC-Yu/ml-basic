import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim

from modules.transformer import Transformer


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


class TestTransformer:
    def __init__(self):
        launcher = TransformerLauncher()

        enc_inputs, dec_inputs, dec_outputs = launcher.make_train_data()
        self.train_loader = launcher.make_dataloader(enc_inputs, dec_inputs, dec_outputs)
        enc_inputs, dec_inputs, dec_outputs = launcher.make_eval_data()
        self.eval_loader = launcher.make_dataloader(enc_inputs, dec_inputs, dec_outputs)

        self.model = launcher.make_transformer()
        self.criterion = nn.CrossEntropyLoss(ignore_index=launcher.padding_index)
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.99)  # 用adam的话效果不好

        self.epochs = launcher.epochs
        self.device = launcher.device
        print(f"Using device: {self.device}")
        self.model.to(self.device)

        self.src_vocab = launcher.src_vocab
        self.tgt_vocab = launcher.tgt_vocab
        self.src_idx2word = launcher.src_idx2word
        self.idx2word = launcher.idx2word
    
    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            for enc_inputs, dec_inputs, dec_outputs in self.train_loader:
                """
                enc_inputs: [batch_size, src_len]
                dec_inputs: [batch_size, tgt_len]
                dec_outputs: [batch_size, tgt_len]
                """
                enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(
                    self.device), dec_inputs.to(self.device), dec_outputs.to(self.device)
                # outputs: [batch_size * tgt_len, tgt_vocab_size]
                outputs, enc_self_attns, dec_self_attns, dec_enc_attns = self.model(
                    enc_inputs, dec_inputs)
                # dec_outputs.view(-1):[batch_size * tgt_len * tgt_vocab_size]
                loss = self.criterion(outputs, dec_outputs.view(-1))
                print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def eval(self):
        self.model.eval()
        enc_inputs, _, _ = next(iter(self.eval_loader))

        print()
        print("="*30)
        print("利用训练好的Transformer模型将中文句子'我 有 零 个 女 朋 友' 翻译成英文句子: ")
        with torch.no_grad():
            for i in range(len(enc_inputs)):
                greedy_dec_predict = greedy_decoder(self.model, enc_inputs[i].view(
                    1, -1).to(self.device), start_symbol=self.tgt_vocab["S"], end_symbol=self.tgt_vocab["E"])
                print(enc_inputs[i], '->', greedy_dec_predict.squeeze())
                print([self.src_idx2word[t.item()] for t in enc_inputs[i]], '->',
                    [self.idx2word[n.item()] for n in greedy_dec_predict.squeeze()])
    

def greedy_decoder(model, enc_input, start_symbol, end_symbol):
    """贪心编码
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    # 初始化一个空的tensor: tensor([], size=(1, 0), dtype=torch.int64)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        # 预测阶段：dec_input序列会一点点变长（每次添加一个新预测出来的单词）
        next_symbol = int(next_symbol)
        dec_input = torch.cat(
            [dec_input, torch.tensor([[next_symbol]], dtype=enc_input.dtype, device=enc_input.device)],
            -1
        )
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.proj(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        # 增量更新（我们希望重复单词预测结果是一样的）
        # 我们在预测是会选择性忽略重复的预测的词，只摘取最新预测的单词拼接到输入序列中
        # 拿出当前预测的单词(数字)。我们用x'_t对应的输出z_t去预测下一个单词的概率，不用z_1,z_2..z_{t-1}
        next_word = prob.data[-1]
        next_symbol = next_word.item()
        if next_symbol == end_symbol:
            terminal = True
        # print(next_word)

    # greedy_dec_predict = torch.cat(
    #     [dec_input.to(device), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(device)],
    #     -1)
    greedy_dec_predict = dec_input[:, 1:]
    return greedy_dec_predict


if __name__ == "__main__":
    tester = TestTransformer()
    tester.train()
    tester.eval()