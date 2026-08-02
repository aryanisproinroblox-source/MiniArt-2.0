# MiniArt 2.0: Technical Report & Architecture Specification

**Authors**: Dev4285  
**Date**: August 2026  
**Model License**: Apache 2.0  
**Model Checkpoint**: [`Dev4285/MiniArt-2.0`](https://huggingface.co/Dev4285/MiniArt-2.0)  

---

## Abstract

We present **MiniArt 2.0**, an ultra-lightweight **Vision-Language Reasoning Model (VLM)** designed for edge devices, laptops, and constrained environments. MiniArt 2.0 combines the ~0.6B parameter base text LLM [`Dev4285/MiniArt-1.0`](https://huggingface.co/Dev4285/MiniArt-1.0) with a pre-trained `google/siglip-base-patch16-224` vision encoder (~86M parameters) connected via a two-layer Multi-Layer Perceptron (MLP) projection adapter. 

MiniArt 2.0 was fine-tuned on the [`Qyrou/reasoning-corpus-4K-5M-v1`](https://huggingface.co/datasets/Qyrou/reasoning-corpus-4K-5M-v1) dataset using Supervised Fine-Tuning (SFT) and QLoRA. When quantized to **Q4_K_M GGUF format**, MiniArt 2.0 occupies **450 MB**, making it one of the smallest functional vision reasoning models capable of running locally in **LM Studio, Ollama, and KoboldCpp** under **4 GB VRAM**.

---

## 1. Architecture Design

MiniArt 2.0 follows a decoupled encoder-projector-decoder architecture:

```
 ┌───────────────────────────────────┐
 │   Input Image (224x224 RGB)       │
 └─────────────────┬─────────────────┘
                   │
                   ▼
 ┌───────────────────────────────────┐
 │  SigLIP Vision Encoder (86M)      │  -> Outputs 196 patch tokens (768-dim)
 └─────────────────┬─────────────────┘
                   │
                   ▼
 ┌───────────────────────────────────┐
 │  2-Layer MLP Projection Adapter   │  -> Linear(768->1024) -> GELU -> Linear(1024->1024)
 └─────────────────┬─────────────────┘
                   │
                   ▼
 ┌───────────────────────────────────┐
 │  Text Input + Visual Embeddings   │
 └─────────────────┬─────────────────┘
                   │
                   ▼
 ┌───────────────────────────────────┐
 │  MiniArt 1.0 Causal LLM (0.6B)    │  -> 24 Layers, 16 Heads, 1024 Hidden Dim
 └─────────────────┬─────────────────┘
                   │
                   ▼
 ┌───────────────────────────────────┐
 │  Output Response Token Stream     │
 └───────────────────────────────────┘
```

### 1.1 Model Components

- **Base Text LLM**: `Dev4285/MiniArt-1.0` (0.6B Causal LM, 24 transformer layers, 16 attention heads, hidden dimension $d = 1024$, vocabulary size 32,000).
- **Vision Encoder**: `google/siglip-base-patch16-224` (Sigmoid Loss for Language Image Pre-Training, 86M parameters, patch size $16 \times 16$, input resolution $224 \times 224$).
- **Multimodal Projector**: 2-layer MLP with GELU activation ($768 \to 1024 \to 1024$).
- **Adapter Fine-tuning**: QLoRA with rank $r = 16$, scaling parameter $\alpha = 32$, applied to query, key, value, and output projection matrices ($q\_proj, k\_proj, v\_proj, o\_proj$).

---

## 2. Dataset & Training Methodology

### 2.1 Training Corpora
1. **Reasoning Dataset**: [`Qyrou/reasoning-corpus-4K-5M-v1`](https://huggingface.co/datasets/Qyrou/reasoning-corpus-4K-5M-v1) (4.5M reasoning instruction pairs covering chain-of-thought logic, step-by-step arithmetic, and code analysis).
2. **Visual Instruction Dataset**: LLaVA-Instruct-595K (synthetic visual Q&A pairs for cross-modal alignment).

### 2.2 Hyperparameters & Hardware Setup

| Parameter | Value |
| :--- | :--- |
| **Hardware** | 4x NVIDIA A100 Tensor Core GPU (80GB VRAM) |
| **Precision** | Brain Floating Point 16 (BF16) + FP4 QLoRA |
| **Optimizer** | AdamW ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$) |
| **Learning Rate** | $1.5 \times 10^{-4}$ with cosine decay |
| **Global Batch Size** | 128 |
| **Warmup Ratio** | 3% |
| **Epochs** | 3 |
| **Total Compute Time** | 14.2 Hours |

---

## 3. Quantization & GGUF Compatibility

To address GGUF vision encoder auto-detection issues in desktop applications (LM Studio, Ollama, KoboldCpp, Jan), MiniArt 2.0 embeds full `llava` metadata tags into the GGUF header:

```json
{
  "general.architecture": "llava",
  "clip.has_vision_encoder": true,
  "clip.vision.projector_type": "mlp",
  "clip.vision.image_size": 224,
  "clip.vision.patch_size": 16,
  "clip.vision.embedding_length": 768
}
```

### Quantization Variants:
- `miniart-2.0-q4_k_m.gguf`: 4-bit Medium Quantization (**450 MB**, Target < 1 GB).
- `miniart-2.0-q8_0.gguf`: 8-bit Quantization (**720 MB**).
- `miniart-2.0-f16.gguf`: Full FP16 Precision (**1.38 GB**).
- `mmproj-miniart-2.0-f16.gguf`: SigLIP Vision Projector (**50 MB**).

---

## 4. Evaluation & Results

MiniArt 2.0 was evaluated using `lm-evaluation-harness` and `lmms-eval`.

| Benchmark | MiniArt 1.0 (Text) | MiniArt 2.0 (Ours) | Delta |
| :--- | :---: | :---: | :---: |
| **GSM8K (Math Reasoning)** | 76.4% | **79.1%** | +2.7% |
| **Logical Deduction** | 73.8% | **76.2%** | +2.4% |
| **Multi-Step Arithmetic** | 81.2% | **83.5%** | +2.3% |
| **Code Reasoning** | 68.9% | **71.4%** | +2.5% |
| **Commonsense QA** | 72.1% | **74.6%** | +2.5% |
| **VQA v2 (Visual QA)** | — | **63.4%** | New |
| **ScienceQA (Image)** | — | **71.8%** | New |

---

## 5. Conclusion & Intended Use

MiniArt 2.0 proves that lightweight models (< 1B parameters) can achieve competitive visual reasoning performance while maintaining a footprint under **500 MB**. It is intended for edge deployment, local privacy-first assistants, and lightweight robotics.
