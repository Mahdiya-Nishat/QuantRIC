import torch
import torch.nn as nn


class TaskDecoder(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.1):
        super().__init__()

        # learnable task query tokens — these specialize during training
        # sensing query: learns to pull sensing-relevant info from fused context
        # comm query:    learns to pull comm-relevant info from fused context
        self.sensing_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.comm_query    = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # sensing branch cross attention
        # query = sensing token, key/value = fused context
        self.sensing_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.sensing_norm = nn.LayerNorm(hidden_dim)
        self.sensing_ffn  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.sensing_norm2 = nn.LayerNorm(hidden_dim)

        # comm branch cross attention
        # query = comm token, key/value = fused context
        self.comm_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.comm_norm  = nn.LayerNorm(hidden_dim)
        self.comm_ffn   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.comm_norm2 = nn.LayerNorm(hidden_dim)

        # output heads
        # sensing head: predicts DoA angle (azimuth, elevation) = 2 values
        # comm head:    predicts beamforming gain = 1 value
        self.sensing_head = nn.Linear(hidden_dim, 2)
        self.comm_head    = nn.Linear(hidden_dim, 1)

    def forward(self, fused):
        # fused: (batch, 256) → expand to sequence for cross attn
        context = fused.unsqueeze(1)   # (batch, 1, 256)

        batch_size = fused.shape[0]

        # --- sensing branch ---
        sq = self.sensing_query.expand(batch_size, -1, -1)  # (batch, 1, 256)
        s_out, s_attn = self.sensing_cross_attn(
            query=sq,
            key=context,
            value=context
        )                                                    # (batch, 1, 256)
        s_out = self.sensing_norm(sq + s_out)               # residual + norm
        s_out = self.sensing_norm2(s_out + self.sensing_ffn(s_out))
        s_out = s_out.squeeze(1)                            # (batch, 256)
        sensing_out = self.sensing_head(s_out)              # (batch, 2)

        # --- comm branch ---
        cq = self.comm_query.expand(batch_size, -1, -1)    # (batch, 1, 256)
        c_out, c_attn = self.comm_cross_attn(
            query=cq,
            key=context,
            value=context
        )                                                   # (batch, 1, 256)
        c_out = self.comm_norm(cq + c_out)                 # residual + norm
        c_out = self.comm_norm2(c_out + self.comm_ffn(c_out))
        c_out = c_out.squeeze(1)                           # (batch, 256)
        comm_out = self.comm_head(c_out)                   # (batch, 1)

        return sensing_out, comm_out, s_attn, c_attn