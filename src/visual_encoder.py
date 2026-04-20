import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import AutoImageProcessor, ViTModel

processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
vit_model.eval()

def generate_scene_image(sample):
    img = Image.new('RGB', (224, 224), color=(0, 0, 0))
    pixels = img.load()

    x_pos  = sample['location'][0]
    y_pos  = sample['location'][1]
    pl     = sample['path_loss']
    los    = sample['los']

    # unique seed per sample so each image looks different
    rng = np.random.default_rng(seed=int(x_pos * 1000 + y_pos))

    # background: signal heatmap grid
    for px in range(224):
        for py in range(224):
            dist = np.sqrt((px - x_pos % 224)**2 + (py - y_pos % 224)**2)
            signal = np.exp(-dist / (pl / 10.0))
            r = int(np.clip(signal * 180 + rng.uniform(0, 30), 0, 255))
            g = int(np.clip(signal * 100 + rng.uniform(0, 20), 0, 255))
            b = int(np.clip((1 - signal) * 200 + rng.uniform(0, 40), 0, 255))
            pixels[px, py] = (r, g, b)

    draw = ImageDraw.Draw(img)

    # user location marker
    ux = int(x_pos % 224)
    uy = int(y_pos % 224)
    color = (0, 255, 80) if los == 1.0 else (255, 40, 40)
    draw.ellipse([ux-8, uy-8, ux+8, uy+8], fill=color, outline=(255,255,255))

    # path loss bar at bottom
    pl_norm = int(np.clip(pl / 150.0 * 200, 0, 200))
    draw.rectangle([12, 205, 12+pl_norm, 218], fill=(80, 160, 255))

    # scatter some obstacle blocks unique to this sample
    n_obstacles = int(rng.uniform(3, 10))
    for _ in range(n_obstacles):
        ox = int(rng.uniform(0, 200))
        oy = int(rng.uniform(0, 200))
        ow = int(rng.uniform(10, 35))
        oh = int(rng.uniform(10, 35))
        gray = int(rng.uniform(60, 140))
        draw.rectangle([ox, oy, ox+ow, oy+oh], fill=(gray, gray, gray))

    return img

def extract_visual_features(sample):
    img = generate_scene_image(sample)
    inputs = processor(images=img, return_tensors='pt')
    with torch.no_grad():
        outputs = vit_model(**inputs)
    full_sequence = outputs.last_hidden_state.squeeze(0)  # (197, 768)
    return full_sequence.numpy()