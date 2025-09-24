# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ---- Load CSV files ----
cosine_avg = pd.read_csv("CosineSimilarity_avg.csv")
cosine_std = pd.read_csv("CosineSimilarity_std.csv")

euclidean_avg = pd.read_csv("EuclideanDistance_Mean_avg.csv")
euclidean_std = pd.read_csv("EuclideanDistance_Std_avg.csv")

hausdorff_avg = pd.read_csv("HausdorffDistance_avg.csv")
hausdorff_std = pd.read_csv("HausdorffDistance_std.csv")

# ---- Dataset-specific plot function with uniform blue color ----
def plot_per_dataset(avg_df, std_df, title_prefix, ylabel, folder):
    datasets = avg_df["Dataset Name"]
    models = avg_df.columns[1:]  # skip "Dataset Name"
    
    # make sure output folder exists
    os.makedirs(folder, exist_ok=True)
    
    for d_idx, dataset in enumerate(datasets):
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(models))  # one bar per model
        
        means = avg_df.loc[d_idx, models]
        errors = std_df.loc[d_idx, models]
        
        ax.bar(
            x, means, yerr=errors, capsize=4,
            color="#87CEEB", edgecolor="black"  # uniform light blue
        )
        
        # Titles and labels
        #ax.set_title(f"{title_prefix} - {dataset}", fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        
        # X-ticks
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=18)
        ax.tick_params(axis="y", labelsize=16)
        
        # Grid
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        
        # Save one file per dataset
        safe_dataset = dataset.replace(" ", "_")
        filename = f"{folder}/{title_prefix.lower().replace(' ', '_')}_{safe_dataset}.png"
        plt.savefig(filename, dpi=700)
        plt.close()

# ---- Generate & save dataset-specific plots ----
plot_per_dataset(cosine_avg, cosine_std, "Cosine Similarity", "Cosine Similarity", "plots/cosine")
plot_per_dataset(euclidean_avg, euclidean_std, "Euclidean Distance", "Euclidean Distance", "plots/euclidean")
plot_per_dataset(hausdorff_avg, hausdorff_std, "Hausdorff Distance", "Hausdorff Distance", "plots/hausdorff")
