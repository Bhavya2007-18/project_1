"""
End-to-End Integrated Text-to-Image Generation Pipeline Controller
"""

import os
import json
import argparse
import text_processor
import cgan_shapes
import attention_gan

def run_pipeline(prompt: str, output_dir="pipeline_outputs"):
    print("=" * 60)
    print(f"[*] Launching Synthesizer Pipeline Core for: '{prompt}'")
    print("=" * 60)
    os.makedirs(output_dir, exist_ok=True)
    
    # Phase 1: Text Embedding Construction
    text_data = text_processor.preprocess_and_embed(prompt)
    
    # Phase 2: Dynamic Conditional Routing
    cgan_labels = ["circle", "square", "triangle"]
    matched_shape = next((shape for shape in cgan_labels if shape in prompt.lower()), None)
    
    img_out_path = os.path.join(output_dir, "synthesized_output.png")
    
    if matched_shape:
        print(f"\n[*] Condition met for basic shapes. Routing to Deep CGAN Core...")
        cgan_shapes.generate_shape(matched_shape, img_out_path)
        mode = "CGAN_Shapes"
    else:
        print(f"\n[*] Complex concept matched. Routing to Self/Cross Attention Generator...")
        img, attention_data = attention_gan.generate_mock_attention_map(prompt, size=256)
        img.save(img_out_path)
        
        with open(os.path.join(output_dir, "attention_maps.json"), "w") as f:
            json.dump(attention_data, f)
        mode = "AttentionGAN"
        
    print(f"\n[*] Execution completed successfully under mode: {mode}")
    print(f"[*] Final Asset exported directly to: {img_out_path}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Command Interface")
    parser.add_argument("--prompt", type=str, default="a crimson crimson rose bloom")
    args = parser.parse_args()
    run_pipeline(args.prompt)