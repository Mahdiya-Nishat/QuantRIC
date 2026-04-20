import torch
import torch.nn as nn

class RadioEncoder(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=256, num_heads=4, num_layers=2, dropout=0.1):
        super(RadioEncoder, self).__init__()

        # each scalar feature becomes a token of size hidden_dim
        self.input_projection = nn.Linear(1, hidden_dim)

        # learnable positional encoding for 7 tokens
        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim, hidden_dim))

        # transformer encoder blocks: MHA + Add&Norm + FFN
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (batch, 7)
        x = x.unsqueeze(-1)                    # (batch, 7, 1)
        x = self.input_projection(x)           # (batch, 7, 256)
        x = x + self.pos_encoding              # add positional encoding
        x = self.transformer(x)                # (batch, 7, 256)
        x = self.norm(x)                       # (batch, 7, 256)
        return x                               # hidden sequence for cross attention