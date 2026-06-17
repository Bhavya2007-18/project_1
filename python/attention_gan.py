"""
Attention-based Generative Adversarial Network Block Definitions
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, C, W, H = x.size()
        N = W * H
        
        proj_query = self.query_conv(x).view(batch, -1, N).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch, -1, N)
        
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        proj_value = self.value_conv(x).view(batch, -1, N)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(batch, C, W, H)
        
        return self.gamma * out + x, attention

class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Conv2d(query_dim, query_dim, kernel_size=1)
        self.key_proj = nn.Linear(key_dim, query_dim)
        self.value_proj = nn.Linear(key_dim, query_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, image_features, word_embeddings):
        batch, C, H, W = image_features.size()
        N = H * W
        
        queries = self.query_proj(image_features).view(batch, C, N).permute(0, 2, 1)
        keys = self.key_proj(word_embeddings)
        values = self.value_proj(word_embeddings)
        
        energy = torch.bmm(queries, keys.permute(0, 2, 1)) / (C ** 0.5)
        attention_map = F.softmax(energy, dim=-1)
        
        context = torch.bmm(attention_map, values).permute(0, 2, 1).view(batch, C, H, W)
        return self.gamma * context + image_features, attention_map

# Procedural Fallback Visualization Generator
def generate_mock_attention_map(prompt: str, size: int = 128):
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np
    
    words = prompt.lower().split()
    composite_img = Image.new("RGB", (size, size), color=(25, 25, 35))
    draw = ImageDraw.Draw(composite_img)
    
    if "rose" in words or "flower" in words:
        draw.ellipse([size//4, size//4, 3*size//4, 3*size//4], fill=(220, 45, 65))
        draw.line([size//2, 3*size//4, size//2, size], fill=(45, 140, 60), width=3)
        
    attention_maps = {}
    for word in words:
        attn_img = Image.new("L", (size, size), color=15)
        a_draw = ImageDraw.Draw(attn_img)
        if word in ["rose", "flower"]:
            a_draw.ellipse([size//4, size//4, 3*size//4, 3*size//4], fill=255)
            attn_img = attn_img.filter(ImageFilter.GaussianBlur(6))
        attention_maps[word] = (np.array(attn_img, dtype=np.float32) / 255.0).tolist()
        
    return composite_img, attention_maps