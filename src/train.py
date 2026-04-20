import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader

from src.radio_encoder import RadioEncoder
from src.visual_encoder import extract_visual_features, generate_scene_image
from src.fusion import VisualProjection, SparseCrossAttentionFusion
from src.task_decoder import TaskDecoder

SCENARIO_PATH = r'C:\Users\Lenovo\PycharmProjects\QuantRIC\deepmimo_scenarios\o1_28b'

# ── dataset ──────────────────────────────────────────────────────────────────
class ISACDataset(Dataset):
    def __init__(self, num_samples=200):
        pl_raw  = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.1.PL.mat')
        loc_raw = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.Loc.mat')
        los_raw = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.1.LoS.mat')

        self.pl_data  = pl_raw['PL_array_full']
        self.loc_data = loc_raw['Loc_array_full']
        self.los_data = los_raw['LOS_tag_array_full']
        self.n        = num_samples

        # normalize path loss to [0,1] for comm label
        pl_vals       = self.pl_data[:num_samples, 0]
        self.pl_min   = pl_vals.min()
        self.pl_max   = pl_vals.max()

        # normalize location x,y for sensing label
        locs          = self.loc_data[:num_samples, :2]
        self.loc_min  = locs.min(axis=0)
        self.loc_max  = locs.max(axis=0)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        loc = self.loc_data[i, :3]
        pl  = self.pl_data[i, 0]
        los = self.los_data[0, i]

        # radio feature vector
        radio = torch.tensor([
            loc[0], loc[1], loc[2],
            pl, los, 0.0, 0.0
        ], dtype=torch.float32)

        # sensing label: normalized x, y
        sensing_label = torch.tensor([
            (loc[0] - self.loc_min[0]) / (self.loc_max[0] - self.loc_min[0] + 1e-8),
            (loc[1] - self.loc_min[1]) / (self.loc_max[1] - self.loc_min[1] + 1e-8),
        ], dtype=torch.float32)

        # comm label: normalized path loss
        comm_label = torch.tensor([
            (pl - self.pl_min) / (self.pl_max - self.pl_min + 1e-8)
        ], dtype=torch.float32)

        # visual features
        sample_dict = {'location': loc, 'path_loss': pl, 'los': los}
        vis = torch.tensor(
            extract_visual_features(sample_dict), dtype=torch.float32
        )  # (197, 768)

        return radio, vis, sensing_label, comm_label


# ── training loop ─────────────────────────────────────────────────────────────
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    dataset    = ISACDataset(num_samples=200)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    radio_encoder = RadioEncoder().to(device)
    visual_proj   = VisualProjection().to(device)
    fusion        = SparseCrossAttentionFusion().to(device)
    task_decoder  = TaskDecoder().to(device)

    params = (
        list(radio_encoder.parameters()) +
        list(visual_proj.parameters()) +
        list(fusion.parameters()) +
        list(task_decoder.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=1e-5)

    sensing_loss_fn = nn.MSELoss()
    comm_loss_fn    = nn.MSELoss()

    epochs = 20
    for epoch in range(epochs):
        total_loss    = 0
        sensing_total = 0
        comm_total    = 0

        for radio, vis, s_label, c_label in dataloader:
            radio   = radio.to(device)    # (B, 7)
            vis     = vis.to(device)      # (B, 197, 768)
            s_label = s_label.to(device)  # (B, 2)
            c_label = c_label.to(device)  # (B, 1)

            optimizer.zero_grad()

            radio_hidden = radio_encoder(radio)   # (B, 7, 256)
            vis_proj     = visual_proj(vis)        # (B, 197, 256)
            fused        = fusion(radio_hidden, vis_proj)  # (B, 256)

            sensing_out, comm_out, _, _ = task_decoder(fused)

            s_loss = sensing_loss_fn(sensing_out, s_label)
            c_loss = comm_loss_fn(comm_out, c_label)
            loss   = s_loss + c_loss

            loss.backward()


            # check where nan comes from
            total_norm = 0
            for p in params:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"  Grad norm: {total_norm:.4f}")

            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            total_loss    += loss.item()
            sensing_total += s_loss.item()
            comm_total    += c_loss.item()

        avg       = total_loss    / len(dataloader)
        avg_s     = sensing_total / len(dataloader)
        avg_c     = comm_total    / len(dataloader)

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Total: {avg:.4f} | "
              f"Sensing: {avg_s:.4f} | "
              f"Comm: {avg_c:.4f}")

    print("\nTraining complete.")
    torch.save({
        'radio_encoder': radio_encoder.state_dict(),
        'visual_proj'  : visual_proj.state_dict(),
        'fusion'       : fusion.state_dict(),
        'task_decoder' : task_decoder.state_dict(),
    }, 'quantric_weights.pth')
    print("Weights saved to quantric_weights.pth")

if __name__ == '__main__':
    train()