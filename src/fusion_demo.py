import torch
import torch.nn as nn

class VisualProjection(nn.Module):
    def __init__(self, vit_dim=768, hidden_dim=256):
        super().__init__()
        self.proj = nn.Linear(vit_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (batch, 197, 768) or (batch, 768)
        return self.norm(self.proj(x))


class SparseCrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.1, top_k=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.top_k      = top_k

        # radio attends to visual
        self.cross_attn_r2v = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # visual attends to radio
        self.cross_attn_v2r = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # FFN after fusion
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, radio_seq, visual_seq):
        # radio_seq:  (batch, 7,   256)
        # visual_seq: (batch, 197, 256)

        # radio queries visual
        r2v, _ = self.cross_attn_r2v(
            query=radio_seq,
            key=visual_seq,
            value=visual_seq
        )
        radio_fused = self.norm1(radio_seq + r2v)   # (batch, 7, 256)

        # visual queries radio
        v2r, _ = self.cross_attn_v2r(
            query=visual_seq,
            key=radio_seq,
            value=radio_seq
        )
        visual_fused = self.norm2(visual_seq + v2r) # (batch, 197, 256)

        # pool both to single vector and add
        radio_pooled  = radio_fused.mean(dim=1)     # (batch, 256)
        visual_pooled = visual_fused.mean(dim=1)    # (batch, 256)

        fused = radio_pooled + visual_pooled         # (batch, 256)
        fused = self.norm3(self.ffn(fused) + fused)  # (batch, 256)

        return fused