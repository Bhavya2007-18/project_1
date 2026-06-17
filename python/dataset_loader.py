"""
Dataset Loader and Analyzer
Loads, structures, and analyzes public text-to-image datasets (e.g., Oxford-102 Flowers).
Computes statistics: class counts, description length distribution, image resolutions,
and saves a structured report.
"""

import os
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTS = True
except ImportError:
    HAS_PLOTS = False

def generate_mock_flowers_dataset(output_dir="dataset_mock", num_samples=150):
    """
    Generates a localized mockup of the Oxford-102 Flowers dataset containing
    images and multiple text description files, mimicking the actual directory structure.
    """
    print(f"[*] Building local mock Oxford-102 Flowers dataset in: {output_dir}")
    os.makedirs(os.path.join(output_dir, "jpg"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "text"), exist_ok=True)
    
    flower_types = ["pink primrose", "wild geranium", "fire lily", "canterbury bells", "globe thistle"]
    descriptions = {
        "pink primrose": [
            "this flower has thin pink petals and a yellow center",
            "a delicate primrose with light pink flowers and yellow pistils",
            "five round pink petals surround a bright yellow center"
        ],
        "wild geranium": [
            "the petals are purple-violet with dark stripes running along them",
            "purple flower with five petals and thin green leaves in background",
            "a small purple wild geranium with white and dark violet veins"
        ],
        "fire lily": [
            "bright orange lily flower with spotted petals curling backwards",
            "vibrant red-orange blossom with long stamens pointing outwards",
            "the orange petals are recurved with dark red spots inside"
        ],
        "canterbury bells": [
            "bell-shaped violet flowers hanging in a cluster from a stem",
            "violet bell flower blooming with white stamens inside",
            "light purple bell-shaped blossoms hanging downwards"
        ],
        "globe thistle": [
            "a round spiky purple flower head on a tall stem",
            "globe thistle with metallic blue spherical flower heads",
            "round ball shaped flower with multiple tiny blue petals and spikes"
        ]
    }
    
    for i in range(num_samples):
        # Pick flower type and index
        cls_idx = i % len(flower_types)
        flower_name = flower_types[cls_idx]
        img_name = f"image_{i+1:04d}.jpg"
        txt_name = f"image_{i+1:04d}.txt"
        
        # 1. Generate procedural flower image
        width, height = np.random.choice([500, 600, 640]), np.random.choice([400, 500, 480])
        img = Image.new("RGB", (width, height), color=(25, 45, 20)) # green-ish dark background
        draw = ImageDraw.Draw(img)
        
        # Draw flower petals
        cx, cy = width // 2, height // 2
        r = 80
        color_map = {
            "pink primrose": (244, 143, 177),
            "wild geranium": (186, 104, 200),
            "fire lily": (255, 112, 67),
            "canterbury bells": (149, 117, 205),
            "globe thistle": (100, 181, 246)
        }
        color = color_map[flower_name]
        
        # Draw petals
        num_petals = 5 if flower_name != "globe thistle" else 12
        for theta in np.linspace(0, 2*np.pi, num_petals, endpoint=False):
            px = cx + int(r * 0.6 * np.cos(theta))
            py = cy + int(r * 0.6 * np.sin(theta))
            draw.ellipse([px-35, py-35, px+35, py+35], fill=color)
            
        # Center of the flower
        center_color = (255, 235, 59) if flower_name == "pink primrose" else (255, 255, 255)
        if flower_name == "globe thistle":
            center_color = color
        draw.ellipse([cx-25, cy-25, cx+25, cy+25], fill=center_color)
        
        # Save image
        img.save(os.path.join(output_dir, "jpg", img_name))
        
        # 2. Save descriptions
        desc_list = descriptions[flower_name]
        # Save with one description per line
        with open(os.path.join(output_dir, "text", txt_name), "w") as f:
            for desc in desc_list:
                f.write(desc + "\n")
                
    print(f"[*] Generated {num_samples} sample files in '{output_dir}'.")

def analyze_dataset(dataset_dir):
    """
    Analyzes images and text annotations. Returns dataset stats.
    """
    jpg_dir = os.path.join(dataset_dir, "jpg")
    txt_dir = os.path.join(dataset_dir, "text")
    
    if not os.path.exists(jpg_dir) or not os.path.exists(txt_dir):
        print("[!] Dataset path not found. Generating mock dataset for analysis.")
        generate_mock_flowers_dataset(dataset_dir)
        
    image_files = [f for f in os.listdir(jpg_dir) if f.endswith(".jpg")]
    text_files = [f for f in os.listdir(txt_dir) if f.endswith(".txt")]
    
    print(f"[*] Analyzing dataset in {dataset_dir}...")
    print(f"Found {len(image_files)} images and {len(text_files)} text annotation files.")
    
    resolutions = []
    description_lengths = []
    all_descriptions = []
    
    for img_name in image_files:
        # Load image size
        img_path = os.path.join(jpg_dir, img_name)
        with Image.open(img_path) as img:
            resolutions.append(img.size)
            
        # Find matching description file
        txt_name = img_name.replace(".jpg", ".txt")
        txt_path = os.path.join(txt_dir, txt_name)
        
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
                for desc in lines:
                    word_count = len(desc.split())
                    description_lengths.append(word_count)
                    all_descriptions.append(desc)
                    
    # Computations
    avg_desc_len = np.mean(description_lengths) if description_lengths else 0
    max_desc_len = np.max(description_lengths) if description_lengths else 0
    min_desc_len = np.min(description_lengths) if description_lengths else 0
    
    resolutions_np = np.array(resolutions)
    widths = resolutions_np[:, 0]
    heights = resolutions_np[:, 1]
    
    unique_resolutions = set(resolutions)
    
    stats = {
        "dataset_name": os.path.basename(dataset_dir),
        "total_images": len(image_files),
        "total_captions": len(all_descriptions),
        "captions_per_image": len(all_descriptions) / len(image_files) if image_files else 0,
        "avg_caption_word_count": float(avg_desc_len),
        "max_caption_word_count": int(max_desc_len),
        "min_caption_word_count": int(min_desc_len),
        "image_resolution_stats": {
            "avg_width": float(np.mean(widths)),
            "avg_height": float(np.mean(heights)),
            "unique_resolutions_count": len(unique_resolutions),
            "common_resolutions": [list(r) for r in list(unique_resolutions)[:5]]
        }
    }
    
    print("\n--- Dataset Analysis Report ---")
    print(f"Total Images: {stats['total_images']}")
    print(f"Total Captions: {stats['total_captions']} (~{stats['captions_per_image']:.1f} per image)")
    print(f"Average Caption Length: {stats['avg_caption_word_count']:.2f} words")
    print(f"Average Image Resolution: {stats['image_resolution_stats']['avg_width']:.1f}x{stats['image_resolution_stats']['avg_height']:.1f}")
    print(f"Unique Resolutions: {stats['image_resolution_stats']['unique_resolutions_count']}")
    
    # Generate Visualizations
    if HAS_PLOTS:
        generate_visualizations(description_lengths, widths, heights, dataset_dir)
    else:
        print("[!] Matplotlib/Seaborn not installed. Skipping visualization generation.")
    
    return stats, all_descriptions

def generate_visualizations(lengths, widths, heights, output_dir):
    """
    Generates and saves visual plots for dataset statistics.
    """
    plots_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"[*] Generating visual outputs in {plots_dir}...")
    
    # 1. Plot Description Length Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(lengths, bins=15, kde=True, color='purple')
    plt.title("Distribution of Caption Word Counts")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "caption_length_dist.png"))
    plt.close()
    
    # 2. Plot Image Resolutions Scatter
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=widths, y=heights, color='teal', alpha=0.7)
    plt.title("Image Resolution Distribution")
    plt.xlabel("Width (px)")
    plt.ylabel("Height (px)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "resolution_scatter.png"))
    plt.close()
    
    print("[*] Visualizations saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Loader & Statistics Analyzer")
    parser.add_argument("--dir", type=str, default="flowers_dataset", help="Directory where dataset is located")
    parser.add_argument("--output", type=str, default="dataset_stats.json", help="Path to save stats JSON")
    args = parser.parse_args()
    
    stats, descriptions = analyze_dataset(args.dir)
    
    # Save statistics report
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[*] Detailed report saved to {args.output}")
