import os
import pandas as pd

# List of dataset names (replace with your actual dataset names)
dataset_names = ["Migraine", "Stroke", "LiverCirrhosis", "LungCancer", "ContraceptiveMethods", "HeartDisease", "HeartFailure", "IndianLiverPatients", "Obesity", "PimaIndianDiabetes"]

# Root directory of the repository
root_dir = os.getcwd()

# Main function to calculate standard deviation
def compute_std_dev():
    for dataset in dataset_names:
        data_frames = []  # To hold data from all t1_ to t5_ folders

        for t_folder in [f"t{i}_EvaluationReport" for i in range(1, 6)]:
            csv_path = os.path.join(root_dir, t_folder, dataset, "supervised", "evaluation_report.csv")
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                data_frames.append(df)
            else:
                print(f"File not found: {csv_path}")

        if data_frames:
            # Concatenate all data frames for this dataset
            combined_df = pd.concat(data_frames, axis=0, ignore_index=True)

            # Group by the 'Model' column and calculate standard deviation for all numeric columns
            std_dev_df = combined_df.groupby("Model").std()

            # Round the standard deviation values to 4 decimal places
            std_dev_df = std_dev_df.round(4)

            # Save the standard deviation data to a new CSV file
            output_dir = os.path.join(root_dir, "std_dev_reports")
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, f"{dataset}_std_dev.csv")
            std_dev_df.to_csv(output_path)

            print(f"Standard deviation report saved for {dataset}: {output_path}")
        else:
            print(f"No data found for dataset {dataset}")

if __name__ == "__main__":
    compute_std_dev()
