import torch
import torch.nn as nn


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
