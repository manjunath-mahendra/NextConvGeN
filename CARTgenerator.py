import numpy as np
import pandas as pd
import json
import os
from synthpop import DataProcessor, CARTMethod


import random
import tensorflow as tf
from tensorflow.keras import backend as K
import keras
import torch

def seed_everything(seed):
    # Python built-in random number generator
    random.seed(seed)
    
    # Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy random number generator
    np.random.seed(seed)
    
    # TensorFlow random number generator
    tf.random.set_seed(seed)
    
    # Keras random number generator
    if tf.__version__.startswith('2'):
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        keras.utils.set_random_seed(seed)
    else:
        tf.compat.v1.keras.backend.clear_session()
        tf.compat.v1.set_random_seed(seed)
        keras.backend.set_random_seed(seed)
    
    # PyTorch random number generator
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # CUDA (GPU) related
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
        #tf.config.experimental.set_seed(seed)


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


def Meta_Data(data, categorical_features, numerical_features):
    column_dict = {}

    for column in data.columns:
        if column in categorical_features:
            column_dict[column] = "categorical"
        elif column in numerical_features:
            column_dict[column] = "numerical"

    return column_dict




def general_pipeline(task, base_directory, synth_directory, balanced_synth_directory, algorithm_name, data_csv_name, action):
    # List all subdirectories (folders) in the base directory
    all_folders = [folder for folder in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, folder))]
    
    # Iterate through all the folders
    for folder in all_folders:
        if folder not in ["Migraine", "Stroke", "LiverCirrhosis", "LungCancer"]:
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
        
        
def CART_action(info, data, training_data_path):
    # Extract information from the info dictionary
    categorical_columns = []
    integer_columns = []

    if info['indices_ordinal_features'] is not None:
        categorical_columns.extend([info['ordered_features'][i] for i in info['indices_ordinal_features']])

    if info['indices_nominal_features'] is not None:
        categorical_columns.extend([info['ordered_features'][i] for i in info['indices_nominal_features']])

    if info['indices_continuous_features'] is not None:
        integer_columns.extend([info['ordered_features'][i] for i in info['indices_continuous_features']])
    
    if info['target'] is not None:
        if isinstance(info.get("target"), str):
            target=info['target']
        else:
            target=info['target'][0]
    
        # Add the target column
        categorical_columns.append(target)
    else:
        target = None

    n_syn_samples=data.shape[0]*5

    metadata = Meta_Data(data, categorical_columns, integer_columns)
    

    processor = DataProcessor(metadata)

    processed_data = processor.preprocess(data)
    
    model = CARTMethod(metadata, smoothing=True, proper=True, minibucket=5, random_state=42)
            
    model.fit(processed_data)
    
    synthetic_processed = model.sample(n_syn_samples)

    seed_everything(42)
    
    syntheticPoints= processor.postprocess(synthetic_processed)

    #syntheticPoints=pd.DataFrame(syntheticPoints, columns=ordered_features)


    # Balance the synthetic data
    balanced_synthetic_data = balance_syn_data(syntheticPoints, info['target_value_counts'], target)
    return syntheticPoints, balanced_synthetic_data



#("NextConvGeN", "NextConvGeN_training_data.csv", NextConvGen_action)
#("CTGAN", "training_data.csv", CTGAN_action),

actions = [("CART", "training_data.csv", CART_action)]

for action in actions:
    #general_pipeline("supervised", "PreparedData1", "t1_SyntheticData", "t1_BalancedSyntheticData", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedData1", "t1_SyntheticData", "t1_BalancedSyntheticData", action[0], action[1], action[2])

    general_pipeline("supervised", "PreparedDataPrivacy1", "t1_SyntheticDataPrivacy", "t1_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedDataPrivacy1", "t1_SyntheticDataPrivacy", "t1_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])

    #general_pipeline("supervised", "PreparedData2", "t2_SyntheticData", "t2_BalancedSyntheticData", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedData2", "t2_SyntheticData", "t2_BalancedSyntheticData", action[0], action[1], action[2])

    general_pipeline("supervised", "PreparedDataPrivacy2", "t2_SyntheticDataPrivacy", "t2_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedDataPrivacy2", "t2_SyntheticDataPrivacy", "t2_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])

    #general_pipeline("supervised", "PreparedData3", "t3_SyntheticData", "t3_BalancedSyntheticData", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedData3", "t3_SyntheticData", "t3_BalancedSyntheticData", action[0], action[1], action[2])

    general_pipeline("supervised", "PreparedDataPrivacy3", "t3_SyntheticDataPrivacy", "t3_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedDataPrivacy3", "t3_SyntheticDataPrivacy", "t3_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])

    #general_pipeline("supervised", "PreparedData4", "t4_SyntheticData", "t4_BalancedSyntheticData", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedData4", "t4_SyntheticData", "t4_BalancedSyntheticData", action[0], action[1], action[2])

    general_pipeline("supervised", "PreparedDataPrivacy4", "t4_SyntheticDataPrivacy", "t4_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedDataPrivacy4", "t4_SyntheticDataPrivacy", "t4_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])

    #general_pipeline("supervised", "PreparedData5", "t5_SyntheticData", "t5_BalancedSyntheticData", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedData5", "t5_SyntheticData", "t5_BalancedSyntheticData", action[0], action[1], action[2])

    general_pipeline("supervised", "PreparedDataPrivacy5", "t5_SyntheticDataPrivacy", "t5_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedDataPrivacy5", "t5_SyntheticDataPrivacy", "t5_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])
