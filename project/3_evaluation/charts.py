import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
RESULTS_DIR = "./"  
SAVE_DIR = "./plots"
USE_LOG_SCALE = False  # <--- TOGGLE FLAG FOR LOGARITHMIC X-AXIS
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- LOAD DATA ----------
rows = []

# Mocking data structure for processing (since files aren't physically present)
# In your local env, this loop stays as you wrote it.
for file in os.listdir(RESULTS_DIR):
    if not file.endswith(".json"): continue
    path = os.path.join(RESULTS_DIR, file)
    try:
        name = file.replace(".json", "")
        method, dataset, checkpoint = name.split("-")
        checkpoint = int(checkpoint)
    except ValueError:
        continue

    with open(path, "r") as f:
        data = json.load(f)
    results = data["results"]
    entity = results["entity_metrics"]

    rows.append({
        "method": method, "dataset": dataset, "checkpoint": checkpoint,
        "rouge1": results["rouge1_f1"], "rouge2": results["rouge2_f1"],
        "rougeL": results["rougeL_f1"], "entity_f1_u": entity["f1_t_u"],
        "entity_f1_nu": entity["f1_t_nu"]
    })

df = pd.DataFrame(rows)
df = df.sort_values(by=["dataset", "method", "checkpoint"])

# ---------- PLOTTING LOGIC ----------
STEPS_PER_EPOCH = 417345 / 5.0  
df["epoch"] = df["checkpoint"] / STEPS_PER_EPOCH

# 1. Consistent Color Palette (Default Matplotlib Cycle)
plt.style.use('ggplot')
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Mapping Metrics to Colors (Ensures ROUGE-1 is always Color A, ROUGE-2 is Color B, etc.)
metric_colors = {
    "rouge1": default_colors[0],       # Blue
    "rouge2": default_colors[1],       # Orange
    "rougeL": default_colors[2],       # Green
    "entity_f1_u": default_colors[3],  # Red
    "entity_f1_nu": default_colors[4]  # Purple
}

# 2. Consistent Markers for Methods
# Mapping Methods to Shapes (LoRA always circle, FFT always square, etc.)
method_markers = {}
available_markers = ["s", "x", "^", "D", "v", "o"]

def get_method_marker(method_name):
    if method_name not in method_markers:
        idx = len(method_markers) % len(available_markers)
        method_markers[method_name] = available_markers[idx]
    return method_markers[method_name]

def plot_separated_metrics():
    rouge_metrics = {"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L"}
    factual_metrics = {"entity_f1_u": "Entity F1 (U)", "entity_f1_nu": "Entity F1 (NU)"}

    for dataset in df["dataset"].unique():
        subset = df[df["dataset"] == dataset]

        for plot_type, metrics_dict in [("rouge", rouge_metrics), ("factual", factual_metrics)]:
            plt.figure(figsize=(10, 6))
            
            for method in subset["method"].unique():
                if (method != "LoRA"):
                    continue
                mdf = subset[subset["method"] == method]
                marker = get_method_marker(method)

                for metric, label in metrics_dict.items():
                    plt.plot(
                        mdf["epoch"], mdf[metric],
                        color=metric_colors[metric],
                        marker=marker,
                        markersize=6,
                        label=f"{label}"
                    )

            plt.title(f"{plot_type.capitalize()} Scores vs Epoch ({method})")
            plt.xlabel("Epoch")
            plt.ylabel("Score (%)")
            plt.ylim(0, 0.5)
            plt.grid(True, which="both", linestyle="--", alpha=0.5)
            
            # 3. Logarithmic Toggle
            if USE_LOG_SCALE:
                plt.xscale('log')
                plt.title(f"{plot_type.capitalize()} Scores vs Epoch (Log Scale)")
            else:
                plt.xticks(range(0, 6))

            plt.legend(fontsize=8, ncol=1, frameon=True)
            plt.savefig(os.path.join(SAVE_DIR, f"{plot_type}_{dataset}.png"), dpi=300)
            plt.close()

plot_separated_metrics()