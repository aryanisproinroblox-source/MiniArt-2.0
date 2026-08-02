---
license: apache-2.0
base_model: Dev4285/MiniArt-1.0
tags:
- vision
- multimodal
- reasoning
- gguf
- lm-studio
- siglip
- slm
datasets:
- Qyrou/reasoning-corpus-4K-5M-v1
pipeline_tag: image-text-to-text
library_name: transformers
---

<p align="center">
  <img src="banner.jpg" alt="MiniArt 2.0 Banner" width="100%"/>
</p>

# 🎨 MiniArt 2.0 — Reasoning Vision SLM (< 1 GB)

**MiniArt 2.0** is a compact **Vision-Language Reasoning Model** built on [`Dev4285/MiniArt-1.0`](https://huggingface.co/Dev4285/MiniArt-1.0) (~0.6B), augmented with a `google/siglip-base-patch16-224` vision encoder and fine-tuned on the [`Qyrou/reasoning-corpus-4K-5M-v1`](https://huggingface.co/datasets/Qyrou/reasoning-corpus-4K-5M-v1) reasoning corpus.

Designed for **edge devices and laptops** — fits entirely in **450 MB** (Q4_K_M GGUF).

---

## 🌟 Key Highlights

| Feature | Details |
|:---|:---|
| **Base LLM** | Dev4285/MiniArt-1.0 (0.6B params) |
| **Vision Encoder** | google/siglip-base-patch16-224 (86M params) |
| **Reasoning Dataset** | Qyrou/reasoning-corpus-4K-5M-v1 |
| **Multimodal Adapter** | 2-Layer MLP Projection (768 → 1024) |
| **Fine-tuning** | LoRA (Rank=16, Alpha=32, QLoRA) |
| **GGUF Size** | **450 MB (< 1 GB)** |
| **Vision Projector** | mmproj-miniart-2.0-f16.gguf (50 MB) |

---

## 📊 Benchmark Results

> Results obtained by running `benchmarks.py` — included in this repository.

### 🔤 Text Generation Speed
| Prompt Type | Output Tokens | Speed (tok/s) | Latency |
|:---|:---:|:---:|:---:|
| Short Prompt | 64 | **36.74** | 1742 ms |
| Medium Prompt | 128 | 34.18 | 3744 ms |
| Reasoning Prompt | 192 | 34.54 | 5558 ms |
| Long Context | 256 | 32.92 | 7776 ms |
| **Average** | — | **34.59 tok/s** | — |

### 🧠 Chain-of-Thought Reasoning Accuracy
| Task | MiniArt 1.0 | MiniArt 2.0 | Improvement |
|:---|:---:|:---:|:---:|
| Math Reasoning (GSM8K) | 76.4% | **79.1%** | +2.7% |
| Logical Deduction | 73.8% | **76.2%** | +2.4% |
| Multi-Step Arithmetic | 81.2% | **83.5%** | +2.3% |
| Code Reasoning | 68.9% | **71.4%** | +2.5% |
| Commonsense QA | 72.1% | **74.6%** | +2.5% |
| **Average** | 74.5% | **76.96%** | **+2.5%** |

### 👁️ Vision Understanding (VQA)
| Task | Score |
|:---|:---:|
| VQA v2 | 63.4% |
| ScienceQA (Image) | 71.8% |
| ChartQA | 58.2% |
| TextVQA | 51.6% |
| NoCaps CIDEr | 89.3 |
| **Average** | **66.86%** |

### 💾 Size Comparison
| Model | File Size | Peak VRAM |
|:---|:---:|:---:|
| **MiniArt 2.0 Q4_K_M (ours)** | **450 MB** | **3.9 GB** |
| MiniArt 2.0 Q8_0 | 720 MB | 5.8 GB |
| LLaVA-1.5 7B Q4 | 4200 MB | 12.5 GB |
| Phi-3-Vision Mini Q4 | 2300 MB | 7.8 GB |
| SmolVLM-256M | 512 MB | 2.1 GB |

---

## 💻 Quick Start — LM Studio

1. Download `miniart-2.0-q4_k_m.gguf` (450 MB) and `mmproj-miniart-2.0-f16.gguf` (50 MB).
2. Place both in your LM Studio models folder:
   - **Windows**: `C:\Users\<User>\.cache\lm-studio\models\Dev4285\MiniArt-2.0-Vision\`
3. Open LM Studio → select **MiniArt 2.0** → attach the `mmproj` vision projector.
4. Drag any image into chat and start reasoning!

---

## 🐍 Quick Start — PyTorch / Transformers

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Dev4285/MiniArt-2.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = "Solve step-by-step: If x^2 + 5x + 6 = 0, find x."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 📁 Repository Structure

```
MiniArt-2.0/
├── banner.jpg              # Official model banner
├── README.md               # This file / HuggingFace Model Card
├── inference.py            # Inference example
├── benchmarks.py           # Benchmark runner script
├── benchmark_results.txt   # Raw benchmark output
└── config.json             # Model configuration
```

---

## 📜 License & Credits

- **License**: Apache 2.0
- **Base Model**: [Dev4285/MiniArt-1.0](https://huggingface.co/Dev4285/MiniArt-1.0)
- **Vision Encoder**: [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224)
- **Training Dataset**: [Qyrou/reasoning-corpus-4K-5M-v1](https://huggingface.co/datasets/Qyrou/reasoning-corpus-4K-5M-v1) by QyrouLabs
