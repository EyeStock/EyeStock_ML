from torchvision.models.video.mvit import PositionalEncoding


class TransformerModel(nn.Module):
    def __init__(self, input_dim=X_FEAT_DIM,
                 d_model=32, nhead=4, num_layers=1, dim_feedforward=64):
        super().__init__()
        # d_model / FF 차원 / layer 수를 전반적으로 축소
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
        # x: (B, 120, 10) -> (120, B, 10)
        x = x.permute(1, 0, 2)           # (S, N, input_dim)
        x = self.input_proj(x)           # (S, N, d_model)
        x = self.pos_encoder(x)          # (S, N, d_model)
        x = self.transformer_encoder(x)  # (S, N, d_model)
        last_step = x[-1]                # (N, d_model)
        out = self.fc(last_step)         # (N, 1)
        return out

    def __init__(self, input_dim=X_FEAT_DIM, d_model=64, nhead=8, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=False,  # 기본 형식 (S, N, E)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, 120, 10) -> (120, B, 10)
        x = x.permute(1, 0, 2)           # (S, N, input_dim)
        x = self.input_proj(x)           # (S, N, d_model)
        x = self.pos_encoder(x)          # (S, N, d_model)
        x = self.transformer_encoder(x)  # (S, N, d_model)
        # 마지막 타임스텝의 representation 사용
        last_step = x[-1]                # (N, d_model)
        out = self.fc(last_step)         # (N, 1)
        return out
