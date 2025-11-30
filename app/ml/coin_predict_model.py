import numpy as np
import torch
import torch.nn as nn

X_FEAT_DIM = 10


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(0)
        return x + self.pe[:seq_len]


class TransformerModel(nn.Module):
    def __init__(self, input_dim=X_FEAT_DIM,
                 d_model=64, nhead=8, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.pos_encoder = PositionalEncoding(d_model)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        last_step = x[-1]
        out = self.fc(last_step)
        return out
