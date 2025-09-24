import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def average_structured_arrays(arrays):
    """
    Average a list of structured arrays with dtype [('Threshold', float), ('Access', float),
    ('Accuracy', float), ('Precision', float)]
    """
    if not arrays:
        return None

    n = len(arrays)
    base = arrays[0]

    avg = np.zeros(base.shape, dtype=base.dtype)
    for field in base.dtype.names:
        stacked = np.stack([arr[field] for arr in arrays], axis=0)  # shape: (runs, entries)
        avg[field] = stacked.mean(axis=0)

    return avg


def average_mia_results(base_dirs, output_base="MIA_average"):
    """
    Average MIA results across multiple runs (t1...t5) and save averaged arrays + plots.
    
    Parameters:
    - base_dirs: list of str, paths like ["t1_PrivacyEvaluationReport", ..., "t5_PrivacyEvaluationReport"]
    - output_base: str, name of folder to save averaged results
    """
    os.makedirs(output_base, exist_ok=True)

    # Collect datasets from the first base_dir
    datasets = os.listdir(base_dirs[0])

    for dataset in datasets:
        supervised_path = os.path.join(base_dirs[0], dataset, "supervised")
        if not os.path.exists(supervised_path):
            continue

        dataset_output_dir = os.path.join(output_base, dataset, "supervised")
        os.makedirs(dataset_output_dir, exist_ok=True)

        # Collect model files from first run
        model_files = [
            f for f in os.listdir(supervised_path)
            if f.endswith("_MIA_scores.npy")
        ]

        for model_file in model_files:
            model_name = model_file.replace("_MIA_scores.npy", "")
            arrays = []

            # Load from all runs
            for base in base_dirs:
                path = os.path.join(base, dataset, "supervised", model_file)
                if os.path.exists(path):
                    arrays.append(np.load(path))
                else:
                    print(f"Warning: {path} not found, skipping")

            if not arrays:
                continue

            # Average structured arrays
            avg_array = average_structured_arrays(arrays)


            # Save averaged array
            save_path = os.path.join(dataset_output_dir, f"{model_name}_MIA_scores_avg.npy")
            np.save(save_path, avg_array)
            print(f"Saved averaged MIA array for {model_name} in {dataset}/supervised")

            # Plot precision vs access
            plot_precision_vs_access(save_path, model_name, dataset_output_dir, plot_name=f"{model_name}_MIA_avg_plot")


def plot_precision_vs_access(npy_path, model_name, output_folder, plot_name="precision_vs_access"):
    structured_array = np.load(npy_path)

    thresholds = np.unique(structured_array['Threshold'])
    colors = ['grey', 'green', 'blue', 'red', 'purple', 'orange']

    plt.figure(figsize=(8, 6))

    for i, t in enumerate(thresholds):
        subset = structured_array[structured_array['Threshold'] == t]
        access_subset = subset['Access']
        precision_subset = subset['Precision']

        plt.plot(
            access_subset,
            precision_subset,
            label=f"Threshold={round(t,1)}",
            linestyle='-',
            marker='o',
            color=colors[i % len(colors)]
        )

    plt.xlabel("Access", fontsize=18)
    plt.ylabel("Precision", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=18)
    plt.title(model_name, fontsize=18)

    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)

    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, f"{plot_name}.png")
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {save_path}")

# ----------------------
# AIA Averaging
# ----------------------
def average_aia_results(base_dirs, output_base="AIA_average"):
    os.makedirs(output_base, exist_ok=True)
    datasets = os.listdir(base_dirs[0])

    for dataset in datasets:
        supervised_path = os.path.join(base_dirs[0], dataset, "supervised")
        if not os.path.exists(supervised_path):
            continue

        all_csvs = []
        for base in base_dirs:
            path = os.path.join(base, dataset, "supervised", "AIA_report.csv")
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    all_csvs.append(df)
                except Exception as e:
                    print(f"Skipping {path}: {e}")

        if not all_csvs:
            continue

        combined = pd.concat(all_csvs)
        summary = combined.groupby("Model").agg(
            f1score_mean=("F1_score", "mean"),
            f1score_std=("F1_score", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std")
        ).reset_index()

        dataset_out_dir = os.path.join(output_base, dataset, "supervised")
        os.makedirs(dataset_out_dir, exist_ok=True)
        save_csv = os.path.join(dataset_out_dir, "AIA_report_avg.csv")
        summary.to_csv(save_csv, index=False)
        print(f"Saved averaged AIA report for {dataset}/supervised -> {save_csv}")

        # Plot Accuracy
        plot_bar(summary, dataset_out_dir, "f1score_mean", "f1score_std",
                 "F1_score", f"{dataset}_AIA_F1_score")

        # Plot RMSE only if valid
        if summary["RMSE_mean"].notna().any():
            plot_bar(summary, dataset_out_dir, "RMSE_mean", "RMSE_std",
                     "RMSE", f"{dataset}_AIA_rmse")


def plot_bar(summary, output_folder, mean_col, std_col, ylabel, filename):
    models = summary["Model"]
    means = summary[mean_col]
    stds = summary[std_col]

    plt.figure(figsize=(10, 6))
    plt.bar(models, means, yerr=stds, capsize=5, color="skyblue")
    plt.ylabel(ylabel, fontsize=20)
    plt.xticks(rotation=45, fontsize=20)
    plt.yticks(fontsize=18)
    #plt.title(f"{ylabel} Comparison", fontsize=18)

    save_path = os.path.join(output_folder, f"{filename}.png")
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved {ylabel} plot -> {save_path}")


# ----------------------
# Run Both
# ----------------------
base_dirs = [
    "t1_PrivacyEvaluationReport",
    "t2_PrivacyEvaluationReport",
    "t3_PrivacyEvaluationReport",
    "t4_PrivacyEvaluationReport",
    "t5_PrivacyEvaluationReport"
]

#average_mia_results(base_dirs, output_base="MIA_average")
average_aia_results(base_dirs, output_base="AIA_average")