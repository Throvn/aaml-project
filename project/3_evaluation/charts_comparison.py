import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
RESULTS_DIR = "./"  
SAVE_DIR = "./plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- LOAD DATA ----------
rows = []

for file in os.listdir(RESULTS_DIR):
    if not file.endswith(".json"): continue
    path = os.path.join(RESULTS_DIR, file)
    
    # Expected filename format: method-dataset-checkpoint.json
    try:
        name = file.replace(".json", "")
        method, dataset, checkpoint = name.split("-")
        checkpoint = int(checkpoint)
    except ValueError:
        print(f"Skipping malformed filename: {file}")
        continue

    with open(path, "r") as f:
        data = json.load(f)
    
    results = data["results"]
    entity = results["entity_metrics"]

    rows.append({
        "method": method, 
        "dataset": dataset, 
        "checkpoint": checkpoint,
        "rouge1": results["rouge1_f1"], 
        "rouge2": results["rouge2_f1"],
        "rougeL": results["rougeL_f1"]
    })

df = pd.DataFrame(rows)

# ---------- COMPARISON LOGIC ----------

def plot_best_method_comparison(df, methods=["FFT", "LoRA"], sort_metric="rougeL"):
    # Filter for the methods we care about
    df_filtered = df[df["method"].isin(methods)]
    
    # Identify the best checkpoint for each method/dataset pair based on sort_metric
    best_idx = df_filtered.groupby(["dataset", "method"])[sort_metric].idxmax()
    best_df = df_filtered.loc[best_idx]

    metrics = ["rouge1", "rouge2", "rougeL"]
    metric_labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    
    # Use default color cycle: FFT will always be color 0, LoRA color 1
    plt.style.use('ggplot')
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    method_colors = {method: default_colors[i] for i, method in enumerate(methods)}

    for dataset in best_df["dataset"].unique():
        subset = best_df[best_df["dataset"] == dataset]
        
        x = np.arange(len(metrics))  # 3 groups for R1, R2, RL
        width = 0.35 
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot bars for each method side-by-side within the metric groups
        for i, method in enumerate(methods):
            method_row = subset[subset["method"] == method]
            if method_row.empty: continue
            
            # Extract scores and convert to 100% scale
            scores = [method_row[m].values[0] * 100 for m in metrics]
            
            # Offset the position so they stand next to each other
            pos = x + (i - len(methods)/2 + 0.5) * width
            
            bars = ax.bar(pos, scores, width, label=method, 
                          color=method_colors[method], edgecolor='white')

            # Add percentage text labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')

        # Formatting
        ax.set_title(f"LoRA vs FFT {dataset}", fontweight='bold', fontsize=14)
        ax.set_ylabel("Score (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontweight='bold')
        ax.set_ylim(0, 55) # Header room for labels
        
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.legend(title="Method", loc='upper right')
        
        plt.tight_layout()
        save_path = os.path.join(SAVE_DIR, f"best_comparison_{dataset}.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
        plt.close()

# Run the analysis
if not df.empty:
    plot_best_method_comparison(df)
else:
    print("No data found to plot.")