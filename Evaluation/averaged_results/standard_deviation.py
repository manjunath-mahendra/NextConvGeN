import pandas as pd
import glob
import os

# Find all metric csv files
files = glob.glob("*_std.csv")

# Dictionary to collect dataset-specific data
datasets = {}

for file in files:
    metric = os.path.basename(file).replace("_std.csv", "")
    df = pd.read_csv(file)

    # First column is dataset name
    dataset_col = df.columns[0]
    for _, row in df.iterrows():
        dataset_name = row[dataset_col]
        # Create dataset-specific table if not exists
        if dataset_name not in datasets:
            datasets[dataset_name] = []
        # Collect row: metric + model scores
        entry = {"metric": metric}
        entry.update(row.drop(dataset_col).to_dict())
        datasets[dataset_name].append(entry)

# Save per dataset
for dataset, rows in datasets.items():
    out_df = pd.DataFrame(rows)
    out_file = f"{dataset}_metrics.csv"
    out_df.to_csv(out_file, index=False)
    print(f"Saved {out_file}")
