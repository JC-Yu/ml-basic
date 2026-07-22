import torch
import torch.nn as nn
import torch.optim as optim
from A_launcher import TransformerLauncher


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
    

def greedy_decoder(self, model, enc_input, start_symbol, end_symbol):
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
        dec_input = torch.cat([dec_input.to(self.device), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(self.device)], -1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        # 增量更新（我们希望重复单词预测结果是一样的）
        # 我们在预测是会选择性忽略重复的预测的词，只摘取最新预测的单词拼接到输入序列中
        # 拿出当前预测的单词(数字)。我们用x'_t对应的输出z_t去预测下一个单词的概率，不用z_1,z_2..z_{t-1}
        next_word = prob.data[-1]
        next_symbol = next_word
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