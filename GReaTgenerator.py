import numpy as np
import pandas as pd

from be_great import GReaT
import json
import os
#import random
#import tensorflow as tf
#from tensorflow.keras import backend as K
#import keras
#import torch



def balance_syn_data(syn_data, value_count, label):
    """
    This function balance the target of synthetic data as that of real data
    
    Args:
        syn_data (DataFrame): Synthetic data from the generative model
        value_count (Dictionary): Target value counts of real data
        label (string): Name of the target column
        
    Returns:
        balanced_sythetic_data (DataFrame): Balanced synthetic data
    
    """
    #columns = syn_data.columns
    df_list = []
    for class_label in value_count:
        if isinstance(class_label, str):
            class_label = float(class_label)
        if isinstance(class_label, float):
            class_label = int(class_label)
            
        class_df = syn_data[syn_data[label] == class_label].sample(n=value_count[str(class_label)], axis=0, random_state=42)
        df_list.append(class_df)
    balanced_synthetic_data = pd.concat(df_list)
    return balanced_synthetic_data.sample(frac=1, random_state=42)


def general_pipeline(task, base_directory, synth_directory, balanced_synth_directory, algorithm_name, data_csv_name, action):
    # List all subdirectories (folders) in the base directory
    all_folders = [folder for folder in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, folder))]
    
    # Iterate through all the folders
    for folder in all_folders:
        if folder in [ "Migraine", "Stroke", "LiverCirrhosis", "LungCancer"]:
            print(folder)
            folder_path = os.path.join(base_directory, folder)

            # Check if the 'semi-supervised' directory exists in the current folder
            task_directory = os.path.join(folder_path, task)
            if not os.path.exists(task_directory):
                print(f"'semi-supervised' directory not found in '{folder}'")
                continue

            # Check if 'training_data.csv' exists in the 'semi-supervised' directory
            training_data_path = os.path.join(task_directory, data_csv_name)
            if not os.path.exists(training_data_path):
                print(f"'{data_csv_name}' not found in '{task_directory}'")
                continue

            # Load the training data
            data = pd.read_csv(training_data_path)

            # Load additional info from the 'additional_info.json' file in the 'semi-supervised' directory
            info_path = os.path.join(task_directory, "additional_info.json")
            if not os.path.exists(info_path):
                print(f"'additional_info.json' file not found in '{task_directory}'")
                continue

            with open(info_path, 'r') as info_file:
                info = json.load(info_file)

            syntheticPoints, balanced_synthetic_data = action(info, data, training_data_path)

            # Create path
            syn_save_path = os.path.join(synth_directory, algorithm_name, folder, task, "synthetic_data.csv")
            balanced_synthetic_data_save_path = os.path.join(balanced_synth_directory, algorithm_name, folder, task, "synthetic_data.csv")

            # Create the directories if they don't exist
            os.makedirs(os.path.dirname(syn_save_path), exist_ok=True)
            os.makedirs(os.path.dirname(balanced_synthetic_data_save_path), exist_ok=True)

            syntheticPoints.to_csv(syn_save_path, index=False)
            balanced_synthetic_data.to_csv(balanced_synthetic_data_save_path, index=False)

            print("Generated Synthetic data for {} and stored in path sucessfully".format(folder))
        
        

def GReaT_action(info, data, training_data_path):
    # Extract information from the info dictionary
    ordered_features = info['ordered_features']

    if info['target'] is not None:
        if isinstance(info.get("target"), str):
            target=info['target']
        else:
            target=info['target'][0]

        # Add the target column
        ordered_features.append(target)
    else:
        target = None

    n_syn_samples=data.shape[0]*5


    model = GReaT(llm='distilgpt2', batch_size=32,  epochs=50,
              logging_steps=50, save_steps=999999, save_total_limit=0)

    model.fit(data)


    
    syntheticPoints= model.sample(n_samples=n_syn_samples, max_length=1000)

    #syntheticPoints=pd.DataFrame(syntheticPoints, columns=ordered_features)


    # Balance the synthetic data
    balanced_synthetic_data = balance_syn_data(syntheticPoints, info['target_value_counts'], target)
    return syntheticPoints, balanced_synthetic_data



#("NextConvGeN", "NextConvGeN_training_data.csv", NextConvGen_action)
#("CTGAN", "training_data.csv", CTGAN_action),

actions = [("GReaT", "training_data.csv", GReaT_action)]

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
