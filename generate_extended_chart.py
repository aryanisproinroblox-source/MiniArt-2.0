#!/usr/bin/env python3
"""
generate_extended_chart.py
Reads /tmp/extended_eval_results.json and generates a comprehensive
multi-benchmark bar chart, then uploads to HuggingFace.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from huggingface_hub import HfApi

HF_TOKEN = os.environ["HF_TOKEN"]
HF_REPO  = "Dev4285/MiniArt-2.0"
api = HfApi(token=HF_TOKEN)

# ── Load results ──────────────────────────────────────────────────────────────
results_path = "/tmp/extended_eval_results.json"
if not os.path.exists(results_path):
    print("No results file found, skipping chart generation")
    exit(0)

with open(results_path) as f:
    results = json.load(f)

# Original 3 benchmarks
original = {
    "gpqa_diamond": 24.2,
    "arc_easy":     56.0,
    "hellaswag":    49.0,
}

# Merge
all_results = {**original, **results}

# Pretty labels
LABELS = {
    "gpqa_diamond":   "GPQA Diamond",
    "arc_easy":       "ARC-Easy",
    "arc_challenge":  "ARC-Challenge",
    "hellaswag":      "HellaSwag",
    "winogrande":     "WinoGrande",
    "piqa":           "PIQA",
    "boolq":          "BoolQ",
    "openbookqa":     "OpenBookQA",
    "truthfulqa_mc1": "TruthfulQA",
    "lambada_openai": "LAMBADA",
    "commonsenseqa":  "CommonsenseQA",
    "copa":           "COPA",
    "rte":            "RTE",
    "wsc":            "WSC",
    "mmlu":           "MMLU",
}

# Random chance baselines
BASELINES = {k: 50.0 if k in ("boolq","copa","rte","wsc","piqa","winogrande","lambada_openai") else 25.0
             for k in all_results}

tasks  = list(all_results.keys())
scores = [all_results[t] for t in tasks]
baselines = [BASELINES.get(t, 25.0) for t in tasks]
labels = [LABELS.get(t, t) for t in tasks]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 8), dpi=160)
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')

x      = np.arange(len(tasks))
width  = 0.38

PALETTE = [
    '#7c3aed','#2563eb','#059669','#dc2626','#d97706',
    '#0891b2','#7c3aed','#be185d','#16a34a','#2563eb',
    '#9333ea','#0d9488','#ea580c','#4f46e5','#0284c7',
]

bars  = ax.bar(x, scores,    width, color=[PALETTE[i % len(PALETTE)] for i in range(len(tasks))],
               edgecolor='none', zorder=3, alpha=0.92)
bbars = ax.bar(x + width, baselines, width, color='#30363d', edgecolor='none', zorder=3, alpha=0.6)

for bar, val in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=8,
            fontweight='bold', color='white')

ax.set_xticks(x + width/2)
ax.set_xticklabels(labels, fontsize=8.5, color='#e6edf3', fontweight='bold', rotation=25, ha='right')
ax.set_ylim(0, 100)
ax.set_ylabel('Accuracy (%)', fontsize=12, color='#8b949e', labelpad=10)
ax.set_title('MiniArt 2.0 — Full Benchmark Suite (15 Tasks)', fontsize=15,
             color='#e6edf3', fontweight='bold', pad=20)

ax.yaxis.set_tick_params(colors='#8b949e')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_color('#30363d')
ax.spines['left'].set_color('#30363d')
ax.yaxis.grid(True, color='#21262d', linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

legend_patches = [mpatches.Patch(color='#7c3aed', label='MiniArt 2.0'),
                  mpatches.Patch(color='#30363d', label='Random Chance')]
ax.legend(handles=legend_patches, loc='upper right', fontsize=10,
          facecolor='#161b22', edgecolor='#30363d', labelcolor='#8b949e')

plt.tight_layout()
chart_path = "/tmp/extended_benchmark_chart.png"
plt.savefig(chart_path, facecolor='#0d1117', dpi=160, bbox_inches='tight')
plt.close()
print(f"Chart saved: {chart_path}")

api.upload_file(
    path_or_fileobj=chart_path,
    path_in_repo="assets/extended_benchmark_chart.png",
    repo_id=HF_REPO,
    repo_type="model",
    commit_message="Add extended 15-task benchmark chart"
)
print("[OK] Extended chart uploaded to HuggingFace!")
