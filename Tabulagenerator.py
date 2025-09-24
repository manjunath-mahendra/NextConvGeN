import numpy as np
import pandas as pd

from tabula import Tabula
import json
import os

from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def balance_syn_data(syn_data, value_count, label):
    """
    Balance the synthetic data to match the target distribution.
    """
    df_list = []

    # Ensure target column is string for consistent comparison
    syn_data[label] = syn_data[label].astype(str)

    for class_label in value_count:
        subset = syn_data[syn_data[label] == class_label]
        needed = value_count[class_label]

        if len(subset) < needed:
            raise ValueError(
                f"Not enough samples for class {class_label}: need {needed}, got {len(subset)}"
            )

        class_df = subset.sample(n=needed, axis=0, random_state=42)
        df_list.append(class_df)

    balanced_synthetic_data = pd.concat(df_list)
    return balanced_synthetic_data.sample(frac=1, random_state=42)




def general_pipeline(task, base_directory, synth_directory, balanced_synth_directory, algorithm_name, data_csv_name, action):
    failed_folders = []  # track skipped datasets

    all_folders = [folder for folder in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, folder))]

    for folder in all_folders:
        if folder not in ["IndianLiverPatients"]:
            try:
                print(folder)
                folder_path = os.path.join(base_directory, folder)
                task_directory = os.path.join(folder_path, task)

                if not os.path.exists(task_directory):
                    print(f"'{task}' directory not found in '{folder}'")
                    continue

                training_data_path = os.path.join(task_directory, data_csv_name)
                if not os.path.exists(training_data_path):
                    print(f"'{data_csv_name}' not found in '{task_directory}'")
                    continue

                data = pd.read_csv(training_data_path)

                info_path = os.path.join(task_directory, "additional_info.json")
                if not os.path.exists(info_path):
                    print(f"'additional_info.json' file not found in '{task_directory}'")
                    continue

                with open(info_path, 'r') as info_file:
                    info = json.load(info_file)

                syntheticPoints, balanced_synthetic_data = action(info, data, training_data_path)

                syn_save_path = os.path.join(synth_directory, algorithm_name, folder, task, "synthetic_data.csv")
                balanced_synthetic_data_save_path = os.path.join(balanced_synth_directory, algorithm_name, folder, task, "synthetic_data.csv")

                os.makedirs(os.path.dirname(syn_save_path), exist_ok=True)
                os.makedirs(os.path.dirname(balanced_synthetic_data_save_path), exist_ok=True)

                syntheticPoints.to_csv(syn_save_path, index=False)
                balanced_synthetic_data.to_csv(balanced_synthetic_data_save_path, index=False)

                print(f"Generated Synthetic data for {folder} and stored successfully")

            except ValueError as e:
                print(f"Skipping {folder} due to balancing error: {e}")
                failed_folders.append(f"{base_directory}/{folder}")
                continue


    if failed_folders:
        failed_log_path = os.path.join(base_directory, f"failed_{algorithm_name}_{task}.txt")
        with open(failed_log_path, "w") as f:
            for name in failed_folders:
                f.write(name + "\n")

        print("\n Skipped folders due to balancing errors:")
        for f in failed_folders:
            print(" -", f)
        print(f"\n Saved skipped dataset names to: {failed_log_path}")


        
        

def Tabula_action(info, data, training_data_path):
    ordered_features = info['ordered_features']

    categorical_columns = []
    integer_columns = []

    if info['indices_ordinal_features'] is not None:
        categorical_columns.extend([ordered_features[i] for i in info['indices_ordinal_features']])

    if info['indices_nominal_features'] is not None:
        categorical_columns.extend([ordered_features[i] for i in info['indices_nominal_features']])

    if info['indices_continuous_features'] is not None:
        integer_columns.extend([ordered_features[i] for i in info['indices_continuous_features']])
    
    if info['target'] is not None:
        if isinstance(info.get("target"), str):
            target = info['target']
        else:
            target = info['target'][0]
        categorical_columns.append(target)
    else:
        target = None

    n_syn_samples = data.shape[0] * 5

    model = Tabula(llm='distilgpt2', batch_size=32, epochs=100, categorical_columns=categorical_columns )
    model.fit(data, conditional_col=ordered_features[0])

    syntheticPoints = model.sample(n_samples=n_syn_samples)

    if not isinstance(syntheticPoints, pd.DataFrame):
        syntheticPoints = pd.DataFrame(syntheticPoints, columns=ordered_features)

    # 🔹 Check if target column exists
    if target not in syntheticPoints.columns:
        print(f"⚠️ Target column '{target}' not found in synthetic data. Skipping balancing for this dataset.")
        return syntheticPoints, syntheticPoints  # return raw synthetic data as fallback

    print("Synthetic data value counts:")
    print(syntheticPoints[target].value_counts())
    print("Real data value counts:")
    print(info['target_value_counts'])

    balanced_synthetic_data = balance_syn_data(syntheticPoints, info['target_value_counts'], target)
    return syntheticPoints, balanced_synthetic_data




#("NextConvGeN", "NextConvGeN_training_data.csv", NextConvGen_action)
#("CTGAN", "training_data.csv", CTGAN_action),

actions = [("Tabula", "training_data.csv", Tabula_action)]

for action in actions:
    general_pipeline("supervised", "PreparedData1", "t1_SyntheticData", "t1_BalancedSyntheticData", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedData1", "t1_SyntheticData", "t1_BalancedSyntheticData", action[0], action[1], action[2])
    print("Completed t1_SyntheticData")
    general_pipeline("supervised", "PreparedDataPrivacy1", "t1_SyntheticDataPrivacy", "t1_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedDataPrivacy1", "t1_SyntheticDataPrivacy", "t1_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])
    print("Completed t1_SyntheticDataPrivacy")
    general_pipeline("supervised", "PreparedData2", "t2_SyntheticData", "t2_BalancedSyntheticData", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedData2", "t2_SyntheticData", "t2_BalancedSyntheticData", action[0], action[1], action[2])
    print("Completed t2_SyntheticData")
    general_pipeline("supervised", "PreparedDataPrivacy2", "t2_SyntheticDataPrivacy", "t2_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedDataPrivacy2", "t2_SyntheticDataPrivacy", "t2_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])
    print("Completed t2_SyntheticDataPrivacy")
    general_pipeline("supervised", "PreparedData3", "t3_SyntheticData", "t3_BalancedSyntheticData", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedData3", "t3_SyntheticData", "t3_BalancedSyntheticData", action[0], action[1], action[2])
    print("Completed t3_SyntheticData")
    general_pipeline("supervised", "PreparedDataPrivacy3", "t3_SyntheticDataPrivacy", "t3_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedDataPrivacy3", "t3_SyntheticDataPrivacy", "t3_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])
    print("Completed t3_SyntheticDataPrivacy")
    general_pipeline("supervised", "PreparedData4", "t4_SyntheticData", "t4_BalancedSyntheticData", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedData4", "t4_SyntheticData", "t4_BalancedSyntheticData", action[0], action[1], action[2])
    print("Completed t4_SyntheticData")
    general_pipeline("supervised", "PreparedDataPrivacy4", "t4_SyntheticDataPrivacy", "t4_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedDataPrivacy4", "t4_SyntheticDataPrivacy", "t4_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])
    print("Completed t4_SyntheticDataPrivacy")  
    general_pipeline("supervised", "PreparedData5", "t5_SyntheticData", "t5_BalancedSyntheticData", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedData5", "t5_SyntheticData", "t5_BalancedSyntheticData", action[0], action[1], action[2])
    print("Completed t5_SyntheticData")
    general_pipeline("supervised", "PreparedDataPrivacy5", "t5_SyntheticDataPrivacy", "t5_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedDataPrivacy5", "t5_SyntheticDataPrivacy", "t5_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])
    print("Completed t5_SyntheticDataPrivacy")
