import numpy as np
import matplotlib.pyplot as plt

# ---------- DATA ----------
models = [
    "T5 base",
    "BART base",
    "PEGASUS large",
    "LLaMA-3-8B",
    "LLaMA-3-8B LoRA",
    "GPT-3.5-turbo",
    "SmolLM base",
    "SmolLM LoRA",
    "SmolLM FFT",
    "SmolLM LoRA (Large Dataset)",
    "SmolLM FFT (Large Dataset)"
]

# Values taken from the supplied latex table in the prompt of chatGPT.

rouge1 = np.array([
    46.84, 46.87, 49.85, 32.92, 45.30, 45.16,
    0.2536983864576781 * 100,
    25.52, 35.36, 34.79, 41.32
])

rouge2 = np.array([
    28.70, 27.66, 30.51, 16.66, 26.53, 23.97,
    0.10783735186395202 * 100,
    10.90, 17.54, 16.84, 22.51
])

rougeL = np.array([
    41.69, 41.89, 43.93, 27.66, 40.51, 38.88,
    0.20466377932509278 * 100,
    20.54, 30.12, 28.61, 35.21
])

# ---------- SORT ----------
sort_idx = np.argsort(-rougeL)

models = [models[i] for i in sort_idx]
rouge1 = rouge1[sort_idx]
rouge2 = rouge2[sort_idx]
rougeL = rougeL[sort_idx]

metrics = [rouge1, rouge2, rougeL]
titles = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]

# ---------- STYLE ----------
plt.style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
bar_colors = (colors * ((len(models) // len(colors)) + 1))[:len(models)]

# Identify SmolLM models
is_smol = np.array(["SmolLM" in m for m in models])

# ---------- PLOT ----------
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

for i, ax in enumerate(axes):
    scores = metrics[i]
    x = np.arange(len(models))

    bars = []
    for j in range(len(models)):
        alpha = 1.0 if is_smol[j] else 0.75  # dim non-SmolLM
        
        bar = ax.bar(
            x[j],
            scores[j],
            color='gray' if not is_smol[j] else bar_colors[j],
            edgecolor='white',
            alpha=alpha
        )
        bars.append(bar[0])

    # Labels
    for j, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            h + 0.6,
            f"{h:.1f}%",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
            alpha=1.0 if is_smol[j] else 0.6
        )

    ax.set_title(titles[i], fontsize=14, fontweight='bold')
    ax.set_ylim(0, 55)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=40, ha='right')

    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].set_ylabel("Score (%)", fontweight='bold')

plt.tight_layout()
plt.savefig("./plots/best_comparison_CSPubSum_all.png", dpi=300)
plt.show()