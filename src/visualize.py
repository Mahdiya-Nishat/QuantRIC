import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from PIL import Image, ImageDraw

from src.radio_encoder import RadioEncoder
from src.visual_encoder import extract_visual_features, generate_scene_image
from src.fusion import VisualProjection, SparseCrossAttentionFusion
from src.task_decoder import TaskDecoder
from src.ris_vqc import RISVQCLayer, generate_ris_labels, train_vqc

SCENARIO_PATH = r'C:\Users\Lenovo\PycharmProjects\QuantRIC\deepmimo_scenarios\o1_28b'

# ── 1. Plot VQC Training Loss ─────────────────────────────────────────────────
def plot_vqc_loss():
    epochs = list(range(1, 11))
    losses = [8.9106, 9.7323, 6.9627, 4.0683, 2.4155,
              1.2761, 0.5303, 0.5105, 0.1346, 0.0174]

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, losses, 'purple', marker='o', linewidth=2, label='RIS Phase Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (log scale)')
    plt.title('VQC Training — RIS Phase Optimization Loss')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('vqc_loss.png', dpi=150)
    plt.show()
    print("Saved vqc_loss.png")


# ── 2. Plot RIS Phase Outputs ─────────────────────────────────────────────────
def plot_ris_phases(num_samples=5):
    device = torch.device('cpu')

    radio_encoder = RadioEncoder().to(device)
    visual_proj   = VisualProjection().to(device)
    fusion        = SparseCrossAttentionFusion().to(device)
    task_decoder  = TaskDecoder().to(device)
    ris_vqc       = RISVQCLayer(hidden_dim=256, num_ris=16).to(device)

    checkpoint = torch.load('quantric_weights.pth', map_location=device)
    radio_encoder.load_state_dict(checkpoint['radio_encoder'])
    visual_proj.load_state_dict(checkpoint['visual_proj'])
    fusion.load_state_dict(checkpoint['fusion'])
    task_decoder.load_state_dict(checkpoint['task_decoder'])
    ris_vqc.load_state_dict(torch.load('vqc_weights.pth', map_location=device))

    for m in [radio_encoder, visual_proj, fusion, task_decoder, ris_vqc]:
        m.eval()

    pl_raw  = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.1.PL.mat')
    loc_raw = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.Loc.mat')
    los_raw = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.1.LoS.mat')

    pl_data  = pl_raw['PL_array_full']
    loc_data = loc_raw['Loc_array_full']
    los_data = los_raw['LOS_tag_array_full']

    fig, axes = plt.subplots(1, num_samples, figsize=(16, 4))
    fig.suptitle('VQC RIS Phase Shifts per Sample (16 Elements)', fontsize=13)

    with torch.no_grad():
        for i in range(num_samples):
            loc = loc_data[i, :3]
            pl  = pl_data[i, 0]
            los = los_data[0, i]

            radio = torch.tensor(
                [loc[0], loc[1], loc[2], pl, los, 0.0, 0.0],
                dtype=torch.float32
            ).unsqueeze(0)

            sample_dict = {'location': loc, 'path_loss': pl, 'los': los}
            vis = torch.tensor(
                extract_visual_features(sample_dict), dtype=torch.float32
            ).unsqueeze(0)

            radio_hidden = radio_encoder(radio)
            vis_proj     = visual_proj(vis)
            fused        = fusion(radio_hidden, vis_proj)
            ris_phases   = ris_vqc(fused).squeeze(0).numpy()  # (16,)
            label        = generate_ris_labels(loc, pl).numpy()

            elements = list(range(1, 17))
            axes[i].plot(elements, label,      'b--o', label='Target',    markersize=4)
            axes[i].plot(elements, ris_phases, 'r-s',  label='VQC Output', markersize=4)
            axes[i].set_title(f'Sample {i}')
            axes[i].set_xlabel('RIS Element')
            axes[i].set_ylabel('Phase (rad)')
            axes[i].set_ylim(0, 2*np.pi)
            axes[i].legend(fontsize=7)
            axes[i].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig('ris_phases.png', dpi=150)
    plt.show()
    print("Saved ris_phases.png")


# ── 3. End to End Inference Demo ──────────────────────────────────────────────
def end_to_end_demo(sample_idx=2):
    device = torch.device('cpu')

    radio_encoder = RadioEncoder().to(device)
    visual_proj   = VisualProjection().to(device)
    fusion        = SparseCrossAttentionFusion().to(device)
    task_decoder  = TaskDecoder().to(device)
    ris_vqc       = RISVQCLayer(hidden_dim=256, num_ris=16).to(device)

    checkpoint = torch.load('quantric_weights.pth', map_location=device)
    radio_encoder.load_state_dict(checkpoint['radio_encoder'])
    visual_proj.load_state_dict(checkpoint['visual_proj'])
    fusion.load_state_dict(checkpoint['fusion'])
    task_decoder.load_state_dict(checkpoint['task_decoder'])
    ris_vqc.load_state_dict(torch.load('vqc_weights.pth', map_location=device))

    for m in [radio_encoder, visual_proj, fusion, task_decoder, ris_vqc]:
        m.eval()

    pl_raw  = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.1.PL.mat')
    loc_raw = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.Loc.mat')
    los_raw = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.1.LoS.mat')

    loc = loc_raw['Loc_array_full'][sample_idx, :3]
    pl  = pl_raw['PL_array_full'][sample_idx, 0]
    los = los_raw['LOS_tag_array_full'][0, sample_idx]

    print("\n" + "="*55)
    print("       QuantRIC — End to End Inference Demo")
    print("="*55)
    print(f"\n[INPUT]")
    print(f"  Sample index : {sample_idx}")
    print(f"  Location     : x={loc[0]:.2f}, y={loc[1]:.2f}, z={loc[2]:.2f}")
    print(f"  Path loss    : {pl:.4f} dB")
    print(f"  LoS status   : {'Line of Sight' if los == 1.0 else 'Non-LoS'}")

    with torch.no_grad():
        # Step 1: Radio encoder
        radio = torch.tensor(
            [loc[0], loc[1], loc[2], pl, los, 0.0, 0.0],
            dtype=torch.float32
        ).unsqueeze(0)
        radio_hidden = radio_encoder(radio)
        print(f"\n[RADIO ENCODER]")
        print(f"  Input  : (1, 7) radio feature vector")
        print(f"  Output : {radio_hidden.shape} hidden sequence")

        # Step 2: Visual encoder
        sample_dict = {'location': loc, 'path_loss': pl, 'los': los}
        img = generate_scene_image(sample_dict)
        img.save(f'data/visual/demo_sample_{sample_idx}.png')
        vis = torch.tensor(
            extract_visual_features(sample_dict), dtype=torch.float32
        ).unsqueeze(0)
        vis_proj = visual_proj(vis)
        print(f"\n[VISUAL ENCODER]")
        print(f"  Input  : synthetic scene image (224x224)")
        print(f"  ViT    : (1, 197, 768) patch sequence")
        print(f"  Output : {vis_proj.shape} projected sequence")

        # Step 3: Cross attention fusion
        fused = fusion(radio_hidden, vis_proj)
        print(f"\n[SPARSE CROSS ATTENTION FUSION]")
        print(f"  Radio  : {radio_hidden.shape}")
        print(f"  Visual : {vis_proj.shape}")
        print(f"  Fused  : {fused.shape} unified scene representation")

        # Step 4: Task decoder
        sensing_out, comm_out, s_attn, c_attn = task_decoder(fused)
        print(f"\n[TASK DECODER]")
        print(f"  Sensing query output : {sensing_out.shape} → {sensing_out.numpy()}")
        print(f"  Comm    query output : {comm_out.shape}    → {comm_out.numpy()}")

        # Step 5: VQC RIS optimization
        ris_phases = ris_vqc(fused).squeeze(0).numpy()
        print(f"\n[VQC RIS OPTIMIZER — 4 qubits x 4 runs = 16 elements]")
        print(f"  RIS phase shifts (rad):")
        for n, phase in enumerate(ris_phases):
            print(f"    Element {n+1:02d}: {phase:.4f} rad  "
                  f"({np.degrees(phase):.1f}°)")

        print(f"\n[RIC DECISION]")
        print(f"  Sensing prediction : azimuth={sensing_out[0,0]:.4f}, "
              f"elevation={sensing_out[0,1]:.4f}")
        print(f"  Comm prediction    : beamforming gain={comm_out[0,0]:.4f}")
        print(f"  RIS configuration  : 16 phase shifts optimized via VQC")
        print("\n" + "="*55)

# ── 4. Multimodal Scene Visualization ────────────────────────────────────────
def plot_multimodal_scene(sample_idx=2):
    device = torch.device('cpu')

    radio_encoder = RadioEncoder().to(device)
    visual_proj   = VisualProjection().to(device)
    fusion        = SparseCrossAttentionFusion().to(device)
    task_decoder  = TaskDecoder().to(device)
    ris_vqc       = RISVQCLayer(hidden_dim=256, num_ris=16).to(device)

    checkpoint = torch.load('quantric_weights.pth', map_location=device)
    radio_encoder.load_state_dict(checkpoint['radio_encoder'])
    visual_proj.load_state_dict(checkpoint['visual_proj'])
    fusion.load_state_dict(checkpoint['fusion'])
    task_decoder.load_state_dict(checkpoint['task_decoder'])
    ris_vqc.load_state_dict(torch.load('vqc_weights.pth', map_location=device))

    for m in [radio_encoder, visual_proj, fusion, task_decoder, ris_vqc]:
        m.eval()

    pl_raw  = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.1.PL.mat')
    loc_raw = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.Loc.mat')
    los_raw = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.1.LoS.mat')

    loc = loc_raw['Loc_array_full'][sample_idx, :3]
    pl  = pl_raw['PL_array_full'][sample_idx, 0]
    los = los_raw['LOS_tag_array_full'][0, sample_idx]

    sample_dict = {'location': loc, 'path_loss': pl, 'los': los}

    # scene image
    img = generate_scene_image(sample_dict)

    # radio heatmap
    grid_size = 50
    heatmap   = np.zeros((grid_size, grid_size))
    cx, cy    = loc[0] % grid_size, loc[1] % grid_size
    for px in range(grid_size):
        for py in range(grid_size):
            dist = np.sqrt((px - cx)**2 + (py - cy)**2)
            heatmap[py, px] = np.exp(-dist / (pl / 20.0))

    # RIS phases
    with torch.no_grad():
        radio  = torch.tensor(
            [loc[0], loc[1], loc[2], pl, los, 0.0, 0.0],
            dtype=torch.float32
        ).unsqueeze(0)
        vis    = torch.tensor(
            extract_visual_features(sample_dict), dtype=torch.float32
        ).unsqueeze(0)
        fused  = fusion(radio_encoder(radio), visual_proj(vis))
        phases = ris_vqc(fused).squeeze(0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'QuantRIC Multimodal Scene — Sample {sample_idx} | '
                 f'PL={pl:.1f}dB | '
                 f'{"LoS" if los==1.0 else "NLoS"}',
                 fontsize=13, fontweight='bold')

    # panel 1: scene image
    axes[0].imshow(img)
    axes[0].set_title('Visual Modality\n(Synthetic UAV Scene)', fontsize=11)
    axes[0].axis('off')

    # panel 2: radio signal heatmap
    im = axes[1].imshow(heatmap, cmap='plasma', origin='lower')
    axes[1].scatter([cx], [cy],
                    color='lime' if los==1.0 else 'red',
                    s=100, zorder=5, label='User')
    axes[1].set_title('Radio Modality\n(Signal Strength Heatmap)', fontsize=11)
    axes[1].set_xlabel('X grid')
    axes[1].set_ylabel('Y grid')
    axes[1].legend(fontsize=9)
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # panel 3: RIS phase wheel
    theta  = np.linspace(0, 2*np.pi, 16, endpoint=False)
    radii  = phases / (2*np.pi)
    ax_pol = fig.add_subplot(1, 3, 3, projection='polar')
    ax_pol.bar(theta, radii, width=2*np.pi/16,
               color=plt.cm.hsv(radii), alpha=0.8, edgecolor='white')
    ax_pol.set_title('RIS Phase Configuration\n(VQC Output)', fontsize=11, pad=15)
    ax_pol.set_yticklabels([])
    axes[2].remove()

    plt.tight_layout()
    plt.savefig('multimodal_scene.png', dpi=150)
    plt.show()
    print("Saved multimodal_scene.png")


# ── 5. RIS Polar Phase Plot ───────────────────────────────────────────────────
def plot_ris_polar(num_samples=5):
    device = torch.device('cpu')

    radio_encoder = RadioEncoder().to(device)
    visual_proj   = VisualProjection().to(device)
    fusion        = SparseCrossAttentionFusion().to(device)
    ris_vqc       = RISVQCLayer(hidden_dim=256, num_ris=16).to(device)

    checkpoint = torch.load('quantric_weights.pth', map_location=device)
    radio_encoder.load_state_dict(checkpoint['radio_encoder'])
    visual_proj.load_state_dict(checkpoint['visual_proj'])
    fusion.load_state_dict(checkpoint['fusion'])
    ris_vqc.load_state_dict(torch.load('vqc_weights.pth', map_location=device))

    for m in [radio_encoder, visual_proj, fusion, ris_vqc]:
        m.eval()

    pl_raw  = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.1.PL.mat')
    loc_raw = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.Loc.mat')
    los_raw = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.1.LoS.mat')

    fig, axes = plt.subplots(1, num_samples,
                             figsize=(18, 4),
                             subplot_kw={'projection': 'polar'})
    fig.suptitle('RIS Phase Shifts — Polar View (16 Elements per Sample)',
                 fontsize=13, fontweight='bold')

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    with torch.no_grad():
        for i in range(num_samples):
            loc = loc_raw['Loc_array_full'][i, :3]
            pl  = pl_raw['PL_array_full'][i, 0]
            los = los_raw['LOS_tag_array_full'][0, i]

            radio = torch.tensor(
                [loc[0], loc[1], loc[2], pl, los, 0.0, 0.0],
                dtype=torch.float32
            ).unsqueeze(0)
            sample_dict = {'location': loc, 'path_loss': pl, 'los': los}
            vis = torch.tensor(
                extract_visual_features(sample_dict), dtype=torch.float32
            ).unsqueeze(0)

            fused  = fusion(radio_encoder(radio), visual_proj(vis))
            phases = ris_vqc(fused).squeeze(0).numpy()

            theta  = np.linspace(0, 2*np.pi, 16, endpoint=False)
            radii  = phases / (2*np.pi)

            axes[i].bar(theta, radii,
                        width=2*np.pi/16,
                        color=colors[i],
                        alpha=0.85,
                        edgecolor='white',
                        linewidth=0.8)
            axes[i].set_title(f'Sample {i}\nPL={pl:.1f}dB',
                              fontsize=9, pad=10)
            axes[i].set_yticklabels([])
            axes[i].set_xticks(theta)
            axes[i].set_xticklabels([str(n+1) for n in range(16)], fontsize=6)

    plt.tight_layout()
    plt.savefig('ris_polar.png', dpi=150)
    plt.show()
    print("Saved ris_polar.png")


# ── 6. Sensing vs Comm Separation Scatter ────────────────────────────────────
def plot_sensing_comm_separation(num_samples=50):
    device = torch.device('cpu')

    radio_encoder = RadioEncoder().to(device)
    visual_proj   = VisualProjection().to(device)
    fusion        = SparseCrossAttentionFusion().to(device)
    task_decoder  = TaskDecoder().to(device)

    checkpoint = torch.load('quantric_weights.pth', map_location=device)
    radio_encoder.load_state_dict(checkpoint['radio_encoder'])
    visual_proj.load_state_dict(checkpoint['visual_proj'])
    fusion.load_state_dict(checkpoint['fusion'])
    task_decoder.load_state_dict(checkpoint['task_decoder'])

    for m in [radio_encoder, visual_proj, fusion, task_decoder]:
        m.eval()

    pl_raw  = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.1.PL.mat')
    loc_raw = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.Loc.mat')
    los_raw = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.1.LoS.mat')

    sensing_vals = []
    comm_vals    = []
    los_flags    = []

    with torch.no_grad():
        for i in range(num_samples):
            loc = loc_raw['Loc_array_full'][i, :3]
            pl  = pl_raw['PL_array_full'][i, 0]
            los = los_raw['LOS_tag_array_full'][0, i]

            radio = torch.tensor(
                [loc[0], loc[1], loc[2], pl, los, 0.0, 0.0],
                dtype=torch.float32
            ).unsqueeze(0)
            sample_dict = {'location': loc, 'path_loss': pl, 'los': los}
            vis = torch.tensor(
                extract_visual_features(sample_dict), dtype=torch.float32
            ).unsqueeze(0)

            fused = fusion(radio_encoder(radio), visual_proj(vis))
            s_out, c_out, _, _ = task_decoder(fused)

            sensing_vals.append(s_out.squeeze(0).numpy())
            comm_vals.append(c_out.squeeze(0).numpy())
            los_flags.append(los)

    sensing_arr = np.array(sensing_vals)   # (50, 2)
    comm_arr    = np.array(comm_vals)      # (50, 1)
    los_arr     = np.array(los_flags)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Sensing vs Comm Branch Separation (Task Decoder Output)',
                 fontsize=13, fontweight='bold')

    # plot 1: sensing output space (azimuth vs elevation)
    los_mask  = los_arr == 1.0
    nlos_mask = ~los_mask
    axes[0].scatter(sensing_arr[los_mask,  0],
                    sensing_arr[los_mask,  1],
                    c='#2ecc71', label='LoS',  s=60, alpha=0.8, edgecolors='k', linewidth=0.5)
    axes[0].scatter(sensing_arr[nlos_mask, 0],
                    sensing_arr[nlos_mask, 1],
                    c='#e74c3c', label='NLoS', s=60, alpha=0.8, edgecolors='k', linewidth=0.5,
                    marker='^')
    axes[0].set_xlabel('Sensing Dim 1 (Azimuth proxy)')
    axes[0].set_ylabel('Sensing Dim 2 (Elevation proxy)')
    axes[0].set_title('Sensing Branch Output Space')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # plot 2: comm output vs sensing magnitude
    sensing_mag = np.linalg.norm(sensing_arr, axis=1)
    axes[1].scatter(sensing_mag[los_mask],
                    comm_arr[los_mask, 0],
                    c='#2ecc71', label='LoS',  s=60, alpha=0.8, edgecolors='k', linewidth=0.5)
    axes[1].scatter(sensing_mag[nlos_mask],
                    comm_arr[nlos_mask, 0],
                    c='#e74c3c', label='NLoS', s=60, alpha=0.8, edgecolors='k', linewidth=0.5,
                    marker='^')
    axes[1].set_xlabel('Sensing Output Magnitude')
    axes[1].set_ylabel('Comm Output (Beamforming Gain proxy)')
    axes[1].set_title('Sensing Magnitude vs Comm Output')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sensing_comm_separation.png', dpi=150)
    plt.show()
    print("Saved sensing_comm_separation.png")
def plot_fused_scene_image(sample_idx=2):
    pl_raw  = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.1.PL.mat')
    loc_raw = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.Loc.mat')
    los_raw = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.1.LoS.mat')

    loc = loc_raw['Loc_array_full'][sample_idx, :3]
    pl  = pl_raw['PL_array_full'][sample_idx, 0]
    los = los_raw['LOS_tag_array_full'][0, sample_idx]

    sample_dict = {'location': loc, 'path_loss': pl, 'los': los}

    # get scene image as numpy
    scene_img = np.array(generate_scene_image(sample_dict)).astype(np.float32)

    # generate radio heatmap at 224x224
    heatmap = np.zeros((224, 224), dtype=np.float32)
    cx = loc[0] % 224
    cy = loc[1] % 224
    for px in range(224):
        for py in range(224):
            dist = np.sqrt((px - cx)**2 + (py - cy)**2)
            heatmap[py, px] = np.exp(-dist / (pl / 10.0))

    # normalize heatmap to 0-255 and colorize with plasma
    heatmap_norm = (heatmap / heatmap.max() * 255).astype(np.uint8)
    cmap    = plt.cm.plasma
    colored = (cmap(heatmap_norm / 255.0)[:, :, :3] * 255).astype(np.float32)

    # blend: 60% scene + 40% radio heatmap
    blended = (0.75 * scene_img + 0.25 * colored).clip(0, 255).astype(np.uint8)
    blended_img = Image.fromarray(blended)

    # draw user location marker on top
    from PIL import ImageDraw
    draw  = ImageDraw.Draw(blended_img)
    ux    = int(cx)
    uy    = int(cy)
    color = (0, 255, 80) if los == 1.0 else (255, 40, 40)
    draw.ellipse([ux-10, uy-10, ux+10, uy+10], fill=color, outline=(255,255,255))
    draw.text((ux+13, uy-8), f'PL={pl:.1f}dB', fill=(255,255,255))

    plt.figure(figsize=(6, 6))
    plt.imshow(blended_img)
    plt.title(f'Radio + Visual Fused Scene — Sample {sample_idx}\n'
              f'{"LoS" if los==1.0 else "NLoS"} | PL={pl:.1f}dB',
              fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('fused_scene.png', dpi=150)
    plt.show()
    print("Saved fused_scene.png")

def plot_rmse_vs_samples():
    device = torch.device('cpu')

    radio_encoder = RadioEncoder().to(device)
    visual_proj   = VisualProjection().to(device)
    fusion        = SparseCrossAttentionFusion().to(device)
    task_decoder  = TaskDecoder().to(device)

    checkpoint = torch.load('quantric_weights.pth', map_location=device)
    radio_encoder.load_state_dict(checkpoint['radio_encoder'])
    visual_proj.load_state_dict(checkpoint['visual_proj'])
    fusion.load_state_dict(checkpoint['fusion'])
    task_decoder.load_state_dict(checkpoint['task_decoder'])

    for m in [radio_encoder, visual_proj, fusion, task_decoder]:
        m.eval()

    pl_raw  = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.1.PL.mat')
    loc_raw = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.Loc.mat')
    los_raw = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.1.LoS.mat')

    pl_data  = pl_raw['PL_array_full']
    loc_data = loc_raw['Loc_array_full']
    los_data = los_raw['LOS_tag_array_full']

    pl_min   = pl_data[:, 0].min()
    pl_max   = pl_data[:, 0].max()
    loc_min  = loc_data[:, :2].min(axis=0)
    loc_max  = loc_data[:, :2].max(axis=0)

    sample_sizes   = [5, 10, 20, 30, 50]
    sensing_rmses  = []
    comm_rmses     = []

    with torch.no_grad():
        for n in sample_sizes:
            s_errs = []
            c_errs = []
            for i in range(n):
                loc = loc_data[i, :3]
                pl  = pl_data[i, 0]
                los = los_data[0, i]

                radio = torch.tensor(
                    [loc[0], loc[1], loc[2], pl, los, 0.0, 0.0],
                    dtype=torch.float32
                ).unsqueeze(0)
                sample_dict = {'location': loc, 'path_loss': pl, 'los': los}
                vis = torch.tensor(
                    extract_visual_features(sample_dict), dtype=torch.float32
                ).unsqueeze(0)

                fused = fusion(radio_encoder(radio), visual_proj(vis))
                s_out, c_out, _, _ = task_decoder(fused)

                s_label = np.array([
                    (loc[0] - loc_min[0]) / (loc_max[0] - loc_min[0] + 1e-8),
                    (loc[1] - loc_min[1]) / (loc_max[1] - loc_min[1] + 1e-8)
                ])
                c_label = np.array([(pl - pl_min) / (pl_max - pl_min + 1e-8)])

                s_errs.append(np.sqrt(np.mean((s_out.numpy() - s_label)**2)))
                c_errs.append(np.sqrt(np.mean((c_out.numpy() - c_label)**2)))

            sensing_rmses.append(np.mean(s_errs))
            comm_rmses.append(np.mean(c_errs))

    # plot 1
    plt.figure(figsize=(8, 5))
    plt.plot(sample_sizes, sensing_rmses, 'b-o', linewidth=2,
             markersize=8, label='Sensing RMSE')
    plt.plot(sample_sizes, comm_rmses,    'r-s', linewidth=2,
             markersize=8, label='Comm RMSE')
    plt.xlabel('Number of Evaluation Samples', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('QuantRIC — RMSE vs Number of Samples', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig('rmse_vs_samples.png', dpi=150)
    plt.show()
    print("Saved rmse_vs_samples.png")


def plot_los_nlos_comparison(num_samples=50):
    device = torch.device('cpu')

    radio_encoder = RadioEncoder().to(device)
    visual_proj   = VisualProjection().to(device)
    fusion        = SparseCrossAttentionFusion().to(device)
    task_decoder  = TaskDecoder().to(device)

    checkpoint = torch.load('quantric_weights.pth', map_location=device)
    radio_encoder.load_state_dict(checkpoint['radio_encoder'])
    visual_proj.load_state_dict(checkpoint['visual_proj'])
    fusion.load_state_dict(checkpoint['fusion'])
    task_decoder.load_state_dict(checkpoint['task_decoder'])

    for m in [radio_encoder, visual_proj, fusion, task_decoder]:
        m.eval()

    pl_raw  = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.1.PL.mat')
    loc_raw = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.Loc.mat')
    los_raw = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.1.LoS.mat')

    pl_data  = pl_raw['PL_array_full']
    loc_data = loc_raw['Loc_array_full']
    los_data = los_raw['LOS_tag_array_full']

    pl_min  = pl_data[:, 0].min()
    pl_max  = pl_data[:, 0].max()
    loc_min = loc_data[:, :2].min(axis=0)
    loc_max = loc_data[:, :2].max(axis=0)

    los_s, los_c   = [], []
    nlos_s, nlos_c = [], []

    with torch.no_grad():
        for i in range(num_samples):
            loc = loc_data[i, :3]
            pl  = pl_data[i, 0]
            los = los_data[0, i]

            radio = torch.tensor(
                [loc[0], loc[1], loc[2], pl, los, 0.0, 0.0],
                dtype=torch.float32
            ).unsqueeze(0)
            sample_dict = {'location': loc, 'path_loss': pl, 'los': los}
            vis = torch.tensor(
                extract_visual_features(sample_dict), dtype=torch.float32
            ).unsqueeze(0)

            fused = fusion(radio_encoder(radio), visual_proj(vis))
            s_out, c_out, _, _ = task_decoder(fused)

            s_label = np.array([
                (loc[0] - loc_min[0]) / (loc_max[0] - loc_min[0] + 1e-8),
                (loc[1] - loc_min[1]) / (loc_max[1] - loc_min[1] + 1e-8)
            ])
            c_label = np.array([(pl - pl_min) / (pl_max - pl_min + 1e-8)])

            s_mse = float(np.mean((s_out.numpy() - s_label)**2))
            c_mse = float(np.mean((c_out.numpy() - c_label)**2))

            if los == 1.0:
                los_s.append(s_mse)
                los_c.append(c_mse)
            else:
                nlos_s.append(s_mse)
                nlos_c.append(c_mse)

    los_sensing_mean  = np.mean(los_s)  if los_s  else 0
    los_comm_mean     = np.mean(los_c)  if los_c  else 0
    nlos_sensing_mean = np.mean(nlos_s) if nlos_s else 0
    nlos_comm_mean    = np.mean(nlos_c) if nlos_c else 0

    print(f"\nLoS  samples : {len(los_s)}")
    print(f"NLoS samples : {len(nlos_s)}")

    # plot 2
    x      = np.array([0, 1])
    width  = 0.3
    s_vals = [los_sensing_mean,  nlos_sensing_mean]
    c_vals = [los_comm_mean,     nlos_comm_mean]

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, s_vals, width, color='#3498db',
            label='Sensing MSE', edgecolor='black', linewidth=0.8)
    plt.bar(x + width/2, c_vals, width, color='#e74c3c',
            label='Comm MSE',    edgecolor='black', linewidth=0.8)
    plt.xticks(x, ['LoS', 'NLoS'], fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title('QuantRIC — LoS vs NLoS Performance Comparison',
              fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig('los_nlos_comparison.png', dpi=150)
    plt.show()
    print("Saved los_nlos_comparison.png")