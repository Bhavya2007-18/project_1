"""
Conditional GAN (CGAN) for Shapes Generation using Deep Convolutional Layers
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageDraw

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

LABEL_MAP = {"circle": 0, "square": 1, "triangle": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def generate_procedural_shape(shape_name: str, size: int = 64, noise_level: float = 0.05):
    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    r = np.random.randint(16, 24)
    
    if shape_name == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    elif shape_name == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=255)
    elif shape_name == "triangle":
        points = [(cx, cy - r), (cx - int(r * 0.866), cy + r // 2), (cx + int(r * 0.866), cy + r // 2)]
        draw.polygon(points, fill=255)
        
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.clip(arr + np.random.normal(0, noise_level, arr.shape), 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8))

def generate_dataset(num_samples=600, size=64):
    images, labels = [], []
    shapes = list(LABEL_MAP.keys())
    for _ in range(num_samples):
        shape = np.random.choice(shapes)
        img = generate_procedural_shape(shape, size)
        arr = np.array(img, dtype=np.float32).reshape(1, size, size) / 127.5 - 1.0
        images.append(arr)
        labels.append(LABEL_MAP[shape])
    return np.array(images), np.array(labels)

if HAS_TORCH:
    class Generator(nn.Module):
        def __init__(self, latent_dim=100, num_classes=3):
            super(Generator, self).__init__()
            self.label_emb = nn.Embedding(num_classes, 50)
            
            self.init_layer = nn.Sequential(nn.Linear(latent_dim + 50, 128 * 16 * 16))
            self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Upsample(scale_factor=2), # 32x32
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2), # 64x64
                nn.Conv2d(64, 1, 3, stride=1, padding=1),
                nn.Tanh()
            )

        def forward(self, noise, labels):
            gen_input = torch.cat((noise, self.label_emb(labels)), -1)
            out = self.init_layer(gen_input)
            out = out.view(-1, 128, 16, 16)
            return self.conv_blocks(out)

    class Discriminator(nn.Module):
        def __init__(self, num_classes=3):
            super(Discriminator, self).__init__()
            self.label_emb = nn.Embedding(num_classes, 64*64)

            self.model = nn.Sequential(
                nn.Conv2d(2, 64, 4, stride=2, padding=1), # 32x32
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
                nn.Conv2d(64, 128, 4, stride=2, padding=1), # 16x16
                nn.BatchNorm2d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
                nn.Flatten(),
                nn.Linear(128 * 16 * 16, 1),
                nn.Sigmoid()
            )

        def forward(self, img, labels):
            lbl_img = self.label_emb(labels).view(-1, 1, 64, 64)
            d_in = torch.cat((img, lbl_img), dim=1)
            return self.model(d_in)

def train_cgan(epochs=5, batch_size=64, lr=0.0002, latent_dim=100):
    if not HAS_TORCH:
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = generate_dataset(num_samples=800)
    dataset = torch.utils.data.TensorDataset(torch.Tensor(images), torch.LongTensor(labels))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    generator = Generator(latent_dim=latent_dim).to(device)
    discriminator = Discriminator().to(device)
    adversarial_loss = nn.BCELoss()
    
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    for epoch in range(epochs):
        for i, (imgs, lbls) in enumerate(dataloader):
            b_size = imgs.size(0)
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            valid = torch.ones(b_size, 1, device=device)
            fake = torch.zeros(b_size, 1, device=device)
            
            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(b_size, latent_dim, device=device)
            gen_labels = torch.randint(0, 3, (b_size,), device=device)
            gen_imgs = generator(z, gen_labels)
            
            g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)
            g_loss.backward()
            optimizer_G.step()
            
            # Train Discriminator
            optimizer_D.zero_grad()
            d_real_loss = adversarial_loss(discriminator(imgs, lbls), valid)
            d_fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
        print(f"Epoch {epoch+1:02d} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
        
    os.makedirs("models", exist_ok=True)
    torch.save(generator.state_dict(), "models/cgan_generator.pth")

def generate_shape(shape_name: str, output_path: str = "generated_shape.png", latent_dim=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if HAS_TORCH and os.path.exists("models/cgan_generator.pth"):
        generator = Generator(latent_dim=latent_dim).to(device)
        generator.load_state_dict(torch.load("models/cgan_generator.pth", map_location=device))
        generator.eval()
        
        z = torch.randn(1, latent_dim, device=device)
        label = torch.LongTensor([LABEL_MAP[shape_name]]).to(device)
        
        with torch.no_grad():
            img_tensor = generator(z, label)[0][0].cpu()
            
        img_np = ((img_tensor.numpy() + 1.0) * 127.5).astype(np.uint8)
        Image.fromarray(img_np).save(output_path)
    else:
        generate_procedural_shape(shape_name).save(output_path)