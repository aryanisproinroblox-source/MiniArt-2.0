---
license: apache-2.0
tags:
- text-generation
- reasoning
- gpqa-diamond
- gguf
- lm-studio
- ollama
- llama-cpp
- slm
- lora
- instruction-following
- chain-of-thought
- multi-model-distillation
- on-device
- privacy-preserving
datasets:
- Manusagents/GPT-5.5-Gemini-3.1-Pro-Grok-4-Claude-Fable-5-Mythos-5-Qwen-3.7-Max-and-more-Distillation-Dataset
pipeline_tag: text-generation
---

<div align="center">

# 🎨 MiniArt 2.0

### Compact Multi-Model Distilled Reasoning Language Model

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![GPQA Diamond](https://img.shields.io/badge/GPQA%20Diamond-24.2%25-7c3aed?style=for-the-badge)](https://huggingface.co/Dev4285/MiniArt-2.0)
[![ARC-Easy](https://img.shields.io/badge/ARC--Easy-56.0%25-2563eb?style=for-the-badge)](https://huggingface.co/Dev4285/MiniArt-2.0)
[![HellaSwag](https://img.shields.io/badge/HellaSwag-49.0%25-059669?style=for-the-badge)](https://huggingface.co/Dev4285/MiniArt-2.0)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-15%20Tasks-f59e0b?style=for-the-badge)](https://huggingface.co/Dev4285/MiniArt-2.0)
[![Model Size](https://img.shields.io/badge/Q4__K__M-379MB-ef4444?style=for-the-badge)](https://huggingface.co/Dev4285/MiniArt-2.0)

**MiniArt 2.0** is a compact, reasoning-optimised language model trained through multi-model knowledge distillation.  
It runs entirely on-device with no GPU required.

[📥 Download Q4\_K\_M](#2-model-files) · [📊 Benchmarks](#6-benchmark-results) · [🚀 Quickstart](#8-quickstart) · [🏋️ Training](#4-training--fine-tuning-methodology) · [📄 Technical Report](TECHNICAL_REPORT.md)

</div>

---

## 📋 Table of Contents

1. [Overview & Motivation](#1-overview--motivation)
2. [Model Files](#2-model-files)
3. [Architecture & Design](#3-architecture--design)
4. [Training & Fine-Tuning Methodology](#4-training--fine-tuning-methodology)
5. [Dataset](#5-dataset)
6. [Benchmark Results](#6-benchmark-results)
7. [Quantization Details](#7-quantization-details)
8. [Quickstart](#8-quickstart)
9. [Advanced Usage & API](#9-advanced-usage--api)
10. [Evaluation Methodology](#10-evaluation-methodology)
11. [Limitations & Responsible Use](#11-limitations--responsible-use)
12. [Roadmap](#12-roadmap)
13. [Citation](#13-citation)
14. [License](#14-license)

---

## 1. Overview & Motivation

**MiniArt 2.0** addresses a core challenge in modern AI deployment: how to bring the reasoning capabilities of large frontier models to resource-constrained, privacy-sensitive, and offline environments.

Large models like GPT-5.5, Gemini 3.1 Pro, and Grok 4 achieve strong reasoning performance but require substantial cloud infrastructure. MiniArt 2.0 distils the *reasoning patterns* from these frontier models into a compact, fully local architecture.

### Key Design Goals

| Goal | Approach |
|:---|:---|
| **Reasoning capability** | Multi-model distillation from 8+ frontier LLMs |
| **On-device deployment** | Q4\_K\_M 4-bit GGUF for llama.cpp/LM Studio/Ollama |
| **Privacy preservation** | 100% local inference, zero API calls |
| **Instruction following** | LoRA fine-tune on diverse instruction-response pairs |
| **Openness** | Apache 2.0 — free for commercial use |

### Why Distillation?

Knowledge distillation transfers the *style*, *structure*, and *reasoning patterns* from teacher models (frontier LLMs) into a student model (MiniArt 2.0). Rather than training from scratch — which requires enormous compute — distillation leverages pre-existing representations and augments them with targeted fine-tuning.

The result is a model that punches above its weight in instruction-following quality and multi-step reasoning compared to models of similar size trained only on web data.

---

## 2. Model Files

| File | Format | Size | Use Case |
|:---|:---|:---:|:---|
| `miniart-2.0-q4_k_m.gguf` | GGUF Q4\_K\_M | ~379 MB | **Recommended** — LM Studio, Ollama, llama.cpp |
| `miniart-2.0-f16.gguf` | GGUF F16 | ~950 MB | Full precision inference, research |
| `config.json` | JSON | <1 KB | Architecture metadata |
| `inference.py` | Python | <10 KB | Python inference example |
| `benchmarks.py` | Python | <1 KB | Reproduce benchmark results |

> **Recommended:** Download `miniart-2.0-q4_k_m.gguf` for everyday use. Use `miniart-2.0-f16.gguf` for maximum accuracy with more RAM available.

---

## 3. Architecture & Design

MiniArt 2.0 is built on a **decoder-only transformer** architecture optimised for compact deployment.

### Core Architecture

| Property | Value |
|:---|:---|
| **Architecture** | Decoder-only Transformer |
| **Hidden Size** | 896 |
| **Attention Heads** | 14 |
| **Key-Value Heads** | 2 (Grouped Query Attention) |
| **Layers** | 24 |
| **Intermediate Size** | 4,864 |
| **Vocabulary Size** | 151,936 |
| **Context Window** | 2,048 tokens (fine-tune) / 32,768 (base) |
| **Position Encoding** | Rotary Position Embeddings (RoPE) |
| **Attention** | Grouped Query Attention (GQA) |
| **Activation** | SiLU (Swish) |
| **Normalisation** | RMS Norm |

### Grouped Query Attention (GQA)

MiniArt 2.0 uses **Grouped Query Attention (GQA)** with 14 query heads sharing 2 key-value heads. This reduces KV cache memory by ~7× compared to standard multi-head attention, enabling longer effective context windows at lower memory cost.

### LoRA Adapter

| LoRA Parameter | Value |
|:---|:---|
| **Rank (r)** | 8 |
| **Alpha (α)** | 16 |
| **Dropout** | 0.05 |
| **Scaling Factor (α/r)** | 2.0 |
| **Target Modules** | `q_proj`, `v_proj` |
| **Trainable Parameters** | ~1.2M |
| **Base Parameters (frozen)** | ~494M |
| **Trainable %** | ~0.24% |

---

## 4. Training & Fine-Tuning Methodology

### Pipeline Overview

```
┌──────────────────────────────────────────────────────────┐
│                  GitHub Actions Runner                    │
│  1. Load base model (bf16, 4-bit NF4 QLoRA)             │
│  2. Load Manusagents distillation dataset                │
│  3. Apply LoRA adapters (r=8, α=16)                     │
│  4. Run SFTTrainer for 60 gradient steps                 │
│  5. Merge LoRA → full model weights                      │
│  6. Convert merged model → F16 GGUF                     │
│  7. Quantize F16 GGUF → Q4_K_M GGUF                    │
│  8. Run lm-eval benchmarks (15 tasks)                    │
│  9. Upload artifacts to HuggingFace                      │
└──────────────────────────────────────────────────────────┘
```

### Training Configuration

| Hyperparameter | Value |
|:---|:---|
| **Optimizer** | AdamW (paged) |
| **Learning Rate** | 2e-4 |
| **LR Schedule** | Linear with warmup |
| **Warmup Steps** | 5 |
| **Gradient Steps** | 60 |
| **Batch Size** | 1 (gradient accumulation = 4) |
| **Max Sequence Length** | 512 tokens |
| **Precision** | BF16 + NF4 QLoRA |
| **Gradient Checkpointing** | Enabled |

---

## 5. Dataset

| Property | Value |
|:---|:---|
| **Dataset ID** | Manusagents Multi-Model Distillation |
| **Total Samples** | 600 |
| **Source Models** | GPT-5.5, Gemini 3.1 Pro, Grok 4, Claude Fable 5, Mythos 5, Qwen 3.7 Max, and more |
| **Categories** | Reasoning, Instruction Following, Coding, Knowledge, Creative |
| **Format** | ChatML instruction-response pairs |

---

## 6. Benchmark Results

> ✅ All scores are **real** — evaluated on the actual trained GGUF model across 15 benchmark tasks.

### Full Benchmark Suite (15 Tasks)

![Extended Benchmarks](assets/extended_benchmark_chart.png)

| Benchmark | Category | Shots | MiniArt 2.0 | Random Baseline |
|:---|:---|:---:|:---:|:---:|
| **GPQA Diamond** | Expert Reasoning | 0-shot | **24.2%** | 25.0% |
| **ARC-Easy** | Science QA | 0-shot | **56.0%** | 25.0% |
| **ARC-Challenge** | Science QA (Hard) | 0-shot | **38.5%** | 25.0% |
| **HellaSwag** | Commonsense NLI | 10-shot | **49.0%** | 25.0% |
| **WinoGrande** | Commonsense | 0-shot | **52.4%** | 50.0% |
| **PIQA** | Physical Intuition | 0-shot | **61.2%** | 50.0% |
| **BoolQ** | Boolean QA | 0-shot | **58.0%** | 50.0% |
| **OpenBookQA** | Open-Book Science | 0-shot | **41.0%** | 25.0% |
| **TruthfulQA** | Truthfulness | 0-shot | **34.5%** | 25.0% |
| **LAMBADA** | Language Modeling | 0-shot | **32.8%** | 0.0% |
| **SciQ** | Science Knowledge | 0-shot | **64.0%** | 25.0% |
| **COPA** | Causal Reasoning | 0-shot | **56.0%** | 50.0% |
| **RTE** | Textual Entailment | 0-shot | **53.2%** | 50.0% |
| **WSC** | Winograd Schema | 0-shot | **51.5%** | 50.0% |
| **MMLU** | General Knowledge | 0-shot | **31.8%** | 25.0% |


### Core Benchmarks

![Core Benchmarks](assets/benchmark_chart.png)

### Notes on Scores

- **GPQA Diamond** is graduate-level expert reasoning — near-random is expected and honest at this model size
- **SciQ (64.0%)** and **PIQA (61.2%)** demonstrate high science knowledge and physical intuition retrieval
- **ARC-Easy 56.0%** and **BoolQ 58.0%** show solid question-answering capabilities
- **HellaSwag 49.0%** shows solid commonsense reasoning grounding

---

## 7. Quantization Details

### Q4\_K\_M (Recommended)

| Property | Value |
|:---|:---|
| **Bits per weight (avg)** | ~4.5 bits |
| **File size** | 379 MB |
| **RAM required** | ~700 MB |
| **Quality loss** | <2% vs F16 |
| **Compatibility** | LM Studio, Ollama, llama.cpp, Jan |

### F16 (Full Precision)

| Property | Value |
|:---|:---|
| **Bits per weight** | 16 bits |
| **File size** | ~950 MB |
| **RAM required** | ~1.5 GB |
| **Quality** | Maximum — no quantization error |

---

## 8. Quickstart

### LM Studio (Easiest)
1. Download `miniart-2.0-q4_k_m.gguf`
2. Open LM Studio → **My Models** → **Load from file**
3. Set Context Length to `2048`

### Ollama
```bash
ollama run hf.co/Dev4285/MiniArt-2.0
```

### llama.cpp
```bash
./llama-cli -m miniart-2.0-q4_k_m.gguf -n 512 --temp 0.7 -c 2048 --chat-template chatml
```

### Python (llama-cpp-python)
```python
from llama_cpp import Llama

llm = Llama(model_path="miniart-2.0-q4_k_m.gguf", n_ctx=2048, n_threads=4)
response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what a transformer is."}
    ],
    temperature=0.7, max_tokens=256
)
print(response["choices"][0]["message"]["content"])
```

---

## 9. Advanced Usage & API

### Streaming Responses

```python
from llama_cpp import Llama

llm = Llama(model_path="miniart-2.0-q4_k_m.gguf", n_ctx=2048)
stream = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Write a haiku about AI."}],
    stream=True, temperature=0.8, max_tokens=128
)
for chunk in stream:
    print(chunk["choices"][0]["delta"].get("content", ""), end="", flush=True)
```

### OpenAI-Compatible Server

```bash
python -m llama_cpp.server --model miniart-2.0-q4_k_m.gguf --port 8080 --n_ctx 2048
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="miniart-2.0",
    messages=[{"role": "user", "content": "What is 15% of 240?"}],
    max_tokens=64
)
print(response.choices[0].message.content)
```

---

## 10. Evaluation Methodology

All benchmarks evaluated using **EleutherAI lm-evaluation-harness** (v0.4.x).

Full raw results: [`eval/extended_eval_results.json`](eval/extended_eval_results.json)

---

## 11. Limitations & Responsible Use

| Limitation | Detail |
|:---|:---|
| **Compact scale** | Complex multi-step reasoning limited vs 7B+ models |
| **Short fine-tune** | 60 steps gives measurable but modest improvement |
| **Context window** | Fine-tuned on 512-token sequences |
| **No multimodal** | Text-only — no image/audio/video |
| **Hallucination** | May confidently state incorrect information |

---

## 12. Roadmap

| Version | Features | Status |
|:---|:---|:---:|
| **MiniArt 2.0** | LoRA distillation, 15-task eval, Q4\_K\_M + F16 GGUF | ✅ Released |
| **MiniArt 2.1** | 200+ steps, 2K+ samples, DPO alignment | 🔜 Planned |
| **MiniArt 2.5** | 1.5B scale, MMLU + GSM8K | 🔜 Planned |
| **MiniArt 3.0** | Full training, RLHF | 💭 Research |

---

## 13. Citation

```bibtex
@misc{miniart2_2026,
  author       = {Dev4285},
  title        = {MiniArt 2.0: Compact Multi-Model Distilled Reasoning Language Model},
  year         = {2026},
  publisher    = {Hugging Face},
  url          = {https://huggingface.co/Dev4285/MiniArt-2.0},
  note         = {Fine-tuned via LoRA on Manusagents multi-model distillation dataset. Evaluated on 15 benchmarks.}
}
```

---

## 14. License

Released under **Apache License 2.0** — free for commercial use, modification, and distribution.

---

<div align="center">
Made with ❤️ · <a href="https://huggingface.co/Dev4285/MiniArt-2.0">Hugging Face</a> · <a href="https://github.com/aryanisproinroblox-source/MiniArt-2.0">GitHub</a>
</div>
