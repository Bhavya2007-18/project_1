# Text-to-Image Generation Pipeline - Internship Project

## Problem Statement
The goal of this project is to construct a comprehensive, modular text-to-image generating pipeline. It bridges the gap between text descriptions and high-fidelity image generation by integrating Conditional GANs (CGANs), Self/Cross-Attention mechanisms, and state-of-the-art text embeddings (Hugging Face Transformers). Additionally, the project explores refining pre-trained models (e.g., Stable Diffusion) on custom domain datasets.

## Dataset
- **Oxford-102 Flowers (Mocked / Localized):** Used for analyzing public dataset structures, visualizing text-description distributions, and image resolutions.
- **Custom Shapes Dataset:** A procedurally generated dataset of basic shapes (circle, square, triangle) to validate the CGAN architecture before scaling up.
- **Custom Art Domain:** Used in the fine-tuning (LoRA) step for adapting pre-trained models like Stable Diffusion to specific art styles.

## Methodology
This project encompasses several core modules implemented as part of the internship tasks:
1. **Dataset Loading & Analysis (`dataset_loader.py`)**: Loads image and text annotations, computes statistics (class counts, description lengths), and provides functions for visualizations (Matplotlib/Seaborn).
2. **Text Preprocessing (`text_processor.py`)**: Utilizes Hugging Face Transformers (`CLIPTokenizer`, `CLIPTextModel`) to tokenize and encode text descriptions into dense global and sequence embeddings.
3. **Attention Strategies (`attention_gan.py`)**: Implements PyTorch-based Self-Attention and Cross-Attention modules that allow the GAN to focus on pertinent features of the text embeddings during generation.
4. **Conditional GANs (`cgan_shapes.py`)**: Implements a deep convolutional CGAN that leverages one-hot encoded textual labels to produce corresponding basic visual concepts.
5. **Pre-trained Model Refinement (`refine_pretrained.py`)**: Implements LoRA (Low-Rank Adaptation) using `peft` and `diffusers` to fine-tune Stable Diffusion UNet layers on custom datasets efficiently.
6. **Integration (`main_pipeline.py`)**: An end-to-end controller that dynamically routes requests based on prompt complexity to either the CGAN or the Attention-GAN block.

## Results & Visual Outputs
- **Dataset Insights:** Detailed JSON reports and visual plots of image resolutions and caption lengths.
- **Attention Maps:** Heatmaps demonstrating the model's focus on specific spatial regions of the image corresponding to the text prompt.
- **CGAN Output:** Accurate generation of geometric shapes matching conditional labels.

## How to Run
Ensure you have the required dependencies (PyTorch, Transformers, Diffusers, PEFT).

1. **Run the Notebook**: Open and execute `python/Internship_Project.ipynb` in Jupyter or Google Colab.
2. **Run the Pipeline CLI**:
   ```bash
   cd python
   python main_pipeline.py --prompt "a beautiful pink primrose"
   ```

## Mentorship & Reproducibility
Code is clean, modular, and thoroughly commented to document preprocessing, feature engineering, and model selection steps, meeting all professional standards.
