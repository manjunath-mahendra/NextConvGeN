import pandas as pd
import os

# Paths for your folders t1...t5
folders = [f"t{i}" for i in range(1, 6)]
filename = "supervisedCombined_report.csv"

# Desired dataset and model orders (with exact renaming)
dataset_order = [
    "Heart failure", "Heart disease", "Lung cancer", "Migraine",
    "Liver cirrhosis", "Indian liver patients", "Pima Indian diabetes",
    "Contraceptive methods", "Obesity", "Stroke"
]
model_order = ["CTGAN", "CTAB-GAN", "TabDDPM",
    "CART", "DataSynth", "GReaT", "NextConvGeN"
]

# Mapping to normalize dataset/model names (extend if needed)
dataset_map = {
    "ContraceptiveMethods": "Contraceptive methods",
    "PimaIndianDiabetes": "Pima Indian diabetes",
    "IndianLiverPatients": "Indian liver patients",
    "HeartFailure":"Heart failure",
    "HeartDisease":"Heart disease",
    "LungCancer":"Lung cancer", 
    "LiverCirrhosis": "Liver cirrhosis", 

}
model_map = {
    "CTABGAN": "CTAB-GAN",
    "CTABGAN+": "CTAB-GAN+",
    "NextConvGeN": "NextConvGeN",
    "Data Synth": "DataSynth",
    "Datasynth": "DataSynth",
    "DataSynthesizer": "DataSynth"
}


# Load and combine all runs
all_dfs = []
for folder in folders:
    path = os.path.join(folder, filename)
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Rename dataset and models for consistency
        df["Dataset Name"] = df["Dataset Name"].replace(dataset_map)
        df["Model"] = df["Model"].replace(model_map)
        all_dfs.append(df)

data = pd.concat(all_dfs, ignore_index=True)

# Identify metric columns
metric_cols = [c for c in data.columns if c not in ["Unnamed: 0", "Dataset Name", "Model"]]

# Group and compute mean/std
grouped = data.groupby(["Dataset Name", "Model"])[metric_cols]
mean_df = grouped.mean().round(4)
std_df = grouped.std().round(4)

# Create separate CSVs per metric
os.makedirs("averaged_results", exist_ok=True)
for metric in metric_cols:
    # Pivot to datasets × models
    avg_pivot = mean_df[metric].reset_index().pivot(
        index="Dataset Name", columns="Model", values=metric
    )
    std_pivot = std_df[metric].reset_index().pivot(
        index="Dataset Name", columns="Model", values=metric
    )

    # Reorder rows/cols explicitly, keep NaN where missing
    avg_pivot = avg_pivot.reindex(index=dataset_order, columns=model_order)
    std_pivot = std_pivot.reindex(index=dataset_order, columns=model_order)

    # Save
    avg_pivot.to_csv(f"averaged_results/{metric}_avg.csv")
    std_pivot.to_csv(f"averaged_results/{metric}_std.csv")
