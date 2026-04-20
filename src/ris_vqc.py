import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio

from src.radio_encoder import RadioEncoder
from src.visual_encoder import extract_visual_features
from src.fusion import VisualProjection, SparseCrossAttentionFusion
from src.task_decoder import TaskDecoder

SCENARIO_PATH = r'C:\Users\Lenovo\PycharmProjects\QuantRIC\deepmimo_scenarios\o1_28b'

# ── 4 qubit VQC circuit (run 4 times → 16 RIS phases) ────────────────────────
dev = qml.device('default.qubit', wires=4)

@qml.qnode(dev, interface='torch')
def vqc_circuit(inputs, weights):
    # inputs:  (4,) — angle encoded features
    # weights: (4, 4) — trainable RY rotations

    # Layer 1: Angle Encoding
    qml.RY(inputs[0], wires=0)
    qml.RY(inputs[1], wires=1)
    qml.RY(inputs[2], wires=2)
    qml.RY(inputs[3], wires=3)

    # Layer 2: Local Entanglement
    qml.RY(weights[0, 0], wires=0)
    qml.RY(weights[0, 1], wires=1)
    qml.RY(weights[0, 2], wires=2)
    qml.RY(weights[0, 3], wires=3)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[2, 3])

    # Layer 3: Cross Branch Coupling
    qml.RY(weights[1, 0], wires=0)
    qml.RY(weights[1, 1], wires=1)
    qml.RY(weights[1, 2], wires=2)
    qml.RY(weights[1, 3], wires=3)
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[3, 0])

    # Layer 4: Global Mixing (ring)
    qml.RY(weights[2, 0], wires=0)
    qml.RY(weights[2, 1], wires=1)
    qml.RY(weights[2, 2], wires=2)
    qml.RY(weights[2, 3], wires=3)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])

    # Final rotation
    qml.RY(weights[3, 0], wires=0)
    qml.RY(weights[3, 1], wires=1)
    qml.RY(weights[3, 2], wires=2)
    qml.RY(weights[3, 3], wires=3)

    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


class RISVQCLayer(nn.Module):
    def __init__(self, hidden_dim=256, num_ris=16):
        super().__init__()
        self.num_ris = num_ris          # 16 RIS elements
        self.num_runs = num_ris // 4    # 4 runs of 4-qubit circuit

        # project fused (256,) to 4 angle inputs per run
        self.input_proj = nn.Linear(hidden_dim, 4 * self.num_runs)

        # separate trainable weights per circuit run
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(4, 4) * 0.01)
            for _ in range(self.num_runs)
        ])

    def forward(self, fused):
        # fused: (batch, 256)
        batch_size = fused.shape[0]

        # project to angle inputs: (batch, 4 * num_runs)
        angles = torch.tanh(self.input_proj(fused)) * torch.pi  # bounded [-pi, pi]

        all_phases = []

        for run in range(self.num_runs):
            run_angles = angles[:, run*4 : run*4+4]  # (batch, 4)
            run_out    = []

            for i in range(batch_size):
                result = vqc_circuit(run_angles[i], self.weights[run])
                result = torch.stack(result).float()  # (4,)
                run_out.append(result)

            run_out = torch.stack(run_out)  # (batch, 4)
            all_phases.append(run_out)

        # concat all runs: (batch, 16)
        ris_phases = torch.cat(all_phases, dim=1)

        # map [-1,1] → [0, 2pi] for RIS unit modulus constraint
        ris_phases = (ris_phases + 1) * torch.pi  # (batch, 16)

        return ris_phases


# ── RIS phase label generator from path loss + location ───────────────────────
def generate_ris_labels(loc, pl, num_ris=16):
    # synthetic optimal phase: based on distance and angle to user
    x, y = loc[0], loc[1]
    angle = np.arctan2(y, x)                          # azimuth angle
    dist  = np.sqrt(x**2 + y**2)
    pl_norm = np.clip(pl / 150.0, 0, 1)

    phases = []
    for n in range(num_ris):
        # each RIS element has slightly different optimal phase
        phi = (angle + n * 2 * np.pi / num_ris + pl_norm) % (2 * np.pi)
        phases.append(phi)

    return torch.tensor(phases, dtype=torch.float32)


# ── Training loop ─────────────────────────────────────────────────────────────
def train_vqc(num_samples=50, epochs=10, batch_size=4):
    device = torch.device('cpu')  # VQC runs on CPU only
    print(f"Training VQC on CPU for {epochs} epochs, {num_samples} samples")

    # load and freeze classical models
    radio_encoder = RadioEncoder().to(device)
    visual_proj   = VisualProjection().to(device)
    fusion        = SparseCrossAttentionFusion().to(device)
    task_decoder  = TaskDecoder().to(device)

    checkpoint = torch.load('quantric_weights.pth', map_location=device)
    radio_encoder.load_state_dict(checkpoint['radio_encoder'])
    visual_proj.load_state_dict(checkpoint['visual_proj'])
    fusion.load_state_dict(checkpoint['fusion'])
    task_decoder.load_state_dict(checkpoint['task_decoder'])

    # freeze all classical weights
    for model in [radio_encoder, visual_proj, fusion, task_decoder]:
        for p in model.parameters():
            p.requires_grad = False

    radio_encoder.eval()
    visual_proj.eval()
    fusion.eval()
    task_decoder.eval()

    # only VQC is trainable
    ris_vqc   = RISVQCLayer(hidden_dim=256, num_ris=16).to(device)
    optimizer = torch.optim.Adam(ris_vqc.parameters(), lr=1e-2)
    loss_fn   = nn.MSELoss()

    # load data
    pl_raw  = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.1.PL.mat')
    loc_raw = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.Loc.mat')
    los_raw = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.1.LoS.mat')

    pl_data  = pl_raw['PL_array_full']
    loc_data = loc_raw['Loc_array_full']
    los_data = los_raw['LOS_tag_array_full']

    for epoch in range(epochs):
        epoch_loss = 0
        steps      = 0

        for start in range(0, num_samples, batch_size):
            end   = min(start + batch_size, num_samples)
            batch_loss = 0

            radio_batch  = []
            vis_batch    = []
            label_batch  = []

            for i in range(start, end):
                loc = loc_data[i, :3]
                pl  = pl_data[i, 0]
                los = los_data[0, i]

                radio_batch.append([loc[0], loc[1], loc[2], pl, los, 0.0, 0.0])

                sample_dict = {'location': loc, 'path_loss': pl, 'los': los}
                vis = extract_visual_features(sample_dict)
                vis_batch.append(vis)

                label_batch.append(generate_ris_labels(loc, pl))

            radio_t = torch.tensor(radio_batch, dtype=torch.float32)
            vis_t   = torch.tensor(np.array(vis_batch), dtype=torch.float32)
            label_t = torch.stack(label_batch)

            with torch.no_grad():
                radio_hidden = radio_encoder(radio_t)
                vis_proj     = visual_proj(vis_t)
                fused        = fusion(radio_hidden, vis_proj)

            optimizer.zero_grad()
            ris_phases = ris_vqc(fused)
            loss       = loss_fn(ris_phases, label_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ris_vqc.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            steps      += 1

        print(f"Epoch {epoch+1:02d}/{epochs} | RIS Phase Loss: {epoch_loss/steps:.4f}")

    torch.save(ris_vqc.state_dict(), 'vqc_weights.pth')
    print("\nVQC weights saved to vqc_weights.pth")
    return ris_vqc