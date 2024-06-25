import numpy as np
from library.generators.XConvGeN import XConvGeN, GeneratorConfig
import pandas as pd
from fdc.fdc import feature_clustering, canberra_modified, Clustering, FDC
from sdv.single_table  import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from model.ctabgan import CTABGAN
import json
import os

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

def meta_data(data, categorical_features, numerical_features):
    column_dict = {}

    for column in data.columns:
        column_dict[column] = {}

        if column in categorical_features:
            column_dict[column]["sdtype"] = "categorical"
        elif column in numerical_features:
            column_dict[column]["sdtype"] = "numerical"
            column_dict[column]["computer_representation"] = "Float"
        else:
            column_dict[column]["sdtype"] = "unknown"

    return {"columns": column_dict}




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
        
        
def TVAE_action(info, data, training_path):
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
        categorical_columns.append(target)
    else:
        target = None

    MetaData = meta_data(data, categorical_columns, integer_columns)
    Meta_Data= SingleTableMetadata.load_from_dict(MetaData)
    
    seed_everything(42)
    # Create the synthesizer with the extracted information
    tvae=TVAESynthesizer(Meta_Data)

    # Fit the synthesizer
    tvae.fit(data)

    # Generate synthetic samples
    n_syn_samples=data.shape[0]*5
    syntheticPoints = tvae.sample(num_rows=n_syn_samples)

    syntheticPoints[categorical_columns]=syntheticPoints[categorical_columns].astype(float).astype(int)

    # Balance the synthetic data
    balanced_synthetic_data = balance_syn_data(syntheticPoints, info['target_value_counts'], target)

    return syntheticPoints, balanced_synthetic_data




def NextConvGen_action(info, data, training_data_path):
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

    fdc = FDC()
    fdc.ord_list=info['indices_ordinal_features']
    if info['indices_nominal_features'] is not None:
        nominal_indices = info['indices_nominal_features']
    else:
        nominal_indices = []

    fdc.nom_list = nominal_indices + [ordered_features.index(target)]

    fdc.cont_list =info['indices_continuous_features']

    train_features=np.array(data)


    seed_everything(42)
    # Train the synthesizer 
    #gen = NextConvGeN(train_features.shape[1], neb=5, fdc=fdc,alpha_clip=0)
    config = GeneratorConfig(n_feat=train_features.shape[1], neb=5, genAddNoise=False, alpha_clip=0.1)
    gen = XConvGeN(config=config, fdc=fdc, debug=False)

    gen.reset(train_features)

    gen.train(train_features)

    syntheticPoints= gen.generateData(n_syn_samples)

    syntheticPoints=pd.DataFrame(syntheticPoints, columns=ordered_features)


    # Balance the synthetic data
    balanced_synthetic_data = balance_syn_data(syntheticPoints, info['target_value_counts'], target)
    return syntheticPoints, balanced_synthetic_data


def CTGAN_action(info, data, training_data_path):
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
    
    MetaData = meta_data(data, categorical_columns, integer_columns)
    Meta_Data= SingleTableMetadata.load_from_dict(MetaData)

    seed_everything(42)
    # Create the synthesizer with the extracted information
    ctgan=CTGANSynthesizer(Meta_Data)

    # Fit the synthesizer
    ctgan.fit(data)

    # Generate synthetic samples
    syntheticPoints = ctgan.sample(num_rows=data.shape[0]*5)
    
    syntheticPoints[categorical_columns]=syntheticPoints[categorical_columns].astype(float).astype(int)

    # Balance the synthetic data
    balanced_synthetic_data = balance_syn_data(syntheticPoints, info['target_value_counts'], target)

    return syntheticPoints, balanced_synthetic_data


def CTABGAN_action(info, data, training_data_path):
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
    seed_everything(42)
    # Create the synthesizer with the extracted information
    synthesizer = CTABGAN(raw_csv_path=training_data_path,
                            categorical_columns=categorical_columns,
                            integer_columns=integer_columns,
                            problem_type={"Classification": target})

    # Fit the synthesizer
    synthesizer.fit()

    # Generate synthetic samples
    syntheticPoints = synthesizer.generate_samples()
    
    syntheticPoints[categorical_columns]=syntheticPoints[categorical_columns].astype(float).astype(int)

    # Balance the synthetic data
    balanced_synthetic_data = balance_syn_data(syntheticPoints, info['target_value_counts'], target)

    return syntheticPoints, balanced_synthetic_data


def CTABGAN_Plus_action(info, data, training_data_path):
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
    seed_everything(42)
    # Create the synthesizer with the extracted information
    synthesizer =  CTABGAN(raw_csv_path = training_data_path, test_ratio = 0.20, categorical_columns = categorical_columns, log_columns = [],
                 mixed_columns= {},
                 non_categorical_columns = [],
                 integer_columns = integer_columns,
                 problem_type= {"Classification": target}) 


    # Fit the synthesizer
    synthesizer.fit()

    # Generate synthetic samples
    syntheticPoints = synthesizer.generate_samples()
    
    syntheticPoints[categorical_columns]=syntheticPoints[categorical_columns].astype(float).astype(int)

    # Balance the synthetic data
    balanced_synthetic_data = balance_syn_data(syntheticPoints, info['target_value_counts'], target)

    return syntheticPoints, balanced_synthetic_data



def reformat_syn_data(task, base_directory, synth_directory, balanced_synth_directory, syn_array_directory = 'TabDDPM_privacy_syn_data'):
    # List all subdirectories (folders) in the base directory
    all_folders = [folder for folder in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, folder))]

    # Iterate through all the folders
    for folder in all_folders:
        folder_path = os.path.join(base_directory, folder)
        syn_array_path = os.path.join(syn_array_directory, folder, task)

        # Check if the 'task' directory exists in the current folder
        task_directory = os.path.join(folder_path, task)
        print(task_directory)
        if os.path.exists(task_directory) and os.path.exists(syn_array_path):
            
            X_num_path = os.path.join(syn_array_path, "X_num_train.npy")
            X_cat_path = os.path.join(syn_array_path, "X_cat_train.npy")
            y_train_path = os.path.join(syn_array_path, "y_train.npy")

            # Load the data
            X_num = np.load(X_num_path, allow_pickle=True)
            X_cat = np.load(X_cat_path, allow_pickle=True)
            y_train = np.load(y_train_path, allow_pickle=True)
            y_train = y_train[:,np.newaxis]

            syn_array = np.concatenate((X_num, X_cat, y_train), axis=1)
            

            # Load additional info from the 'additional_info.json' file in the 'semi-supervised' directory
            info_path = os.path.join(task_directory, "additional_info.json")
            if os.path.exists(info_path):
                with open(info_path, 'r') as info_file:
                    info = json.load(info_file)

                # Extract information from the info dictionary
                ordered_features = info['ordered_features']

                if isinstance(info.get("target"), str):
                    target=info['target']
                else:
                    target=info['target'][0]

                # Add the target column
                if info['target'] is not None:
                    ordered_features.append(target)
                
                syn_df = pd.DataFrame(syn_array, columns = ordered_features)


                # Balance the synthetic data
                balanced_synthetic_data = balance_syn_data(syn_df, info['target_value_counts'], target)

                # Create path
                syn_save_path = os.path.join(synth_directory,"TabDDPM", folder, task, "synthetic_data.csv")
                balanced_synthetic_data_save_path = os.path.join(balanced_synth_directory,"TabDDPM", folder, task, "synthetic_data.csv")

                # Create the directories if they don't exist
                os.makedirs(os.path.dirname(syn_save_path), exist_ok=True)
                os.makedirs(os.path.dirname(balanced_synthetic_data_save_path), exist_ok=True)

                syn_df.to_csv(syn_save_path, index=False)
                balanced_synthetic_data.to_csv(balanced_synthetic_data_save_path, index=False)

                print("Formatted Synthetic data for {} and stored in path sucessfully".format(folder))

            else:
                print(f"Directory not found in '{task_directory}'")

        else:
            print(f"Directory not found in '{folder}'")



#("NextConvGeN", "NextConvGeN_training_data.csv", NextConvGen_action)
#("CTGAN", "training_data.csv", CTGAN_action),

actions = [#("NextConvGeN", "NextConvGeN_training_data.csv", NextConvGen_action), 
            #("convexCTGAN", "synthetic_data.csv", CTGAN_action),
            #("convexCTABGAN", "synthetic_data.csv", CTABGAN_action),
            #("CTABGANplus", "training_data.csv", CTABGAN_Plus_action),
            ("TVAE", "training_data.csv", TVAE_action)]

for action in actions:
    general_pipeline("supervised", "PreparedData1", "t1_SyntheticData", "t1_BalancedSyntheticData", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedData1", "t1_SyntheticData", "t1_BalancedSyntheticData", action[0], action[1], action[2])

    #general_pipeline("supervised", "PreparedDataPrivacy1", "t1_SyntheticDataPrivacy", "t1_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedDataPrivacy1", "t1_SyntheticDataPrivacy", "t1_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])

    general_pipeline("supervised", "PreparedData2", "t2_SyntheticData", "t2_BalancedSyntheticData", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedData2", "t2_SyntheticData", "t2_BalancedSyntheticData", action[0], action[1], action[2])

    #general_pipeline("supervised", "PreparedDataPrivacy2", "t2_SyntheticDataPrivacy", "t2_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedDataPrivacy2", "t2_SyntheticDataPrivacy", "t2_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])

    general_pipeline("supervised", "PreparedData3", "t3_SyntheticData", "t3_BalancedSyntheticData", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedData3", "t3_SyntheticData", "t3_BalancedSyntheticData", action[0], action[1], action[2])

    #general_pipeline("supervised", "PreparedDataPrivacy3", "t3_SyntheticDataPrivacy", "t3_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedDataPrivacy3", "t3_SyntheticDataPrivacy", "t3_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])

    general_pipeline("supervised", "PreparedData4", "t4_SyntheticData", "t4_BalancedSyntheticData", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedData4", "t4_SyntheticData", "t4_BalancedSyntheticData", action[0], action[1], action[2])

    #general_pipeline("supervised", "PreparedDataPrivacy4", "t4_SyntheticDataPrivacy", "t4_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedDataPrivacy4", "t4_SyntheticDataPrivacy", "t4_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])

    general_pipeline("supervised", "PreparedData5", "t5_SyntheticData", "t5_BalancedSyntheticData", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedData5", "t5_SyntheticData", "t5_BalancedSyntheticData", action[0], action[1], action[2])

    #general_pipeline("supervised", "PreparedDataPrivacy5", "t5_SyntheticDataPrivacy", "t5_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])
    #general_pipeline("semi-supervised", "PreparedDataPrivacy5", "t5_SyntheticDataPrivacy", "t5_BalancedSyntheticDataPrivacy", action[0], action[1], action[2])

"""
reformat_syn_data("supervised", "PreparedDataPrivacy1", "t1_SyntheticDataPrivacy","t1_BalancedSyntheticDataPrivacy", syn_array_directory = 'TabDDPM_privacy_output1')
reformat_syn_data("semi-supervised", "PreparedDataPrivacy1", "t1_SyntheticDataPrivacy","t1_BalancedSyntheticDataPrivacy", syn_array_directory = 'TabDDPM_output1')

reformat_syn_data("supervised", "PreparedDataPrivacy2", "t2_SyntheticDataPrivacy","t2_BalancedSyntheticDataPrivacy", syn_array_directory = 'TabDDPM_output2')
reformat_syn_data("semi-supervised", "PreparedDataPrivacy2", "t2_SyntheticDataPrivacy","t2_BalancedSyntheticDataPrivacy", syn_array_directory = 'TabDDPM_output2')

reformat_syn_data("supervised", "PreparedDataPrivacy3", "t3_SyntheticDataPrivacy","t3_BalancedSyntheticDataPrivacy", syn_array_directory = 'TabDDPM_output3')
reformat_syn_data("semi-supervised", "PreparedDataPrivacy3", "t3_SyntheticDataPrivacy","t3_BalancedSyntheticDataPrivacy", syn_array_directory = 'TabDDPM_output3')

reformat_syn_data("supervised", "PreparedDataPrivacy4", "t4_SyntheticDataPrivacy","t4_BalancedSyntheticDataPrivacy", syn_array_directory = 'TabDDPM_output4')
reformat_syn_data("semi-supervised", "PreparedDataPrivacy4", "t4_SyntheticDataPrivacy","t4_BalancedSyntheticDataPrivacy", syn_array_directory = 'TabDDPM_output4')

reformat_syn_data("supervised", "PreparedDataPrivacy5", "t5_SyntheticDataPrivacy","t5_BalancedSyntheticDataPrivacy", syn_array_directory = 'TabDDPM_output5')
reformat_syn_data("semi-supervised", "PreparedDataPrivacy5", "t5_SyntheticDataPrivacy","t5_BalancedSyntheticDataPrivacy", syn_array_directory = 'TabDDPM_output5')
"""

    

#=================================================================================================================
          
def TVAE_pipeline(task, base_directory, synth_directory, balanced_synth_directory):
    # List all subdirectories (folders) in the base directory
    all_folders = [folder for folder in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, folder))]

    # Iterate through all the folders
    for folder in all_folders:
        folder_path = os.path.join(base_directory, folder)

        # Check if the 'semi-supervised' directory exists in the current folder
        task_directory = os.path.join(folder_path, task)
        if os.path.exists(task_directory):

            # Check if 'training_data.csv' exists in the 'semi-supervised' directory
            training_data_path = os.path.join(task_directory, "training_data.csv")
            if os.path.exists(training_data_path):

                # Load the training data
                data = pd.read_csv(training_data_path)

                # Load additional info from the 'additional_info.json' file in the 'semi-supervised' directory
                info_path = os.path.join(task_directory, "additional_info.json")
                if os.path.exists(info_path):
                    with open(info_path, 'r') as info_file:
                        info = json.load(info_file)

                    # Extract information from the info dictionary
                    categorical_columns = []
                    integer_columns = []

                    if info['indices_ordinal_features'] is not None:
                        categorical_columns.extend([info['ordered_features'][i] for i in info['indices_ordinal_features']])

                    if info['indices_nominal_features'] is not None:
                        categorical_columns.extend([info['ordered_features'][i] for i in info['indices_nominal_features']])

                    if info['indices_continuous_features'] is not None:
                        integer_columns.extend([info['ordered_features'][i] for i in info['indices_continuous_features']])
                    
                    if isinstance(info.get("target"), str):
                        target=info['target']
                    else:
                        target=info['target'][0]
                    
                    # Add the target column
                    if info['target'] is not None:
                        categorical_columns.append(target)
                    
                    MetaData = meta_data(data, categorical_columns, integer_columns)
                    Meta_Data= SingleTableMetadata.load_from_dict(MetaData)


                    # Create the synthesizer with the extracted information
                    tvae=TVAESynthesizer(Meta_Data)

                    # Fit the synthesizer
                    tvae.fit(data)

                    # Generate synthetic samples
                    n_syn_samples=data.shape[0]*5
                    syn = tvae.sample(num_rows=n_syn_samples)
                    
                    syn[categorical_columns]=syn[categorical_columns].astype(float).astype(int)

                    # Balance the synthetic data
                    #balanced_synthetic_data = balance_syn_data(syntheticPoints, info['target_value_counts'], target)
                    max_attempts = 5  # Adjust this as needed
                    attempts = 0

                    while attempts < max_attempts:
                        try:
                            # Attempt to balance the synthetic data
                            balanced_synthetic_data = balance_syn_data(syn, info['target_value_counts'], target)
                            break  # Break out of the loop if successful
                        except ValueError as e:
                            # Handle the ValueError (sampling size larger than population) here
                            print(f"Error: {e}")

                            # Adjust the number of synthetic samples and regenerate data
                            n_syn_samples *= 2  
                            syn = tvae.sample(num_rows=n_syn_samples)

                            #syn = pd.DataFrame(syntheticPoints, columns=ordered_features)

                            attempts += 1

                    if attempts == max_attempts:
                        print("Maximum attempts reached. Error not resolved.")
                    else:
                        print("Balancing successful after", attempts, "attempts.")

                    # Create path
                    syn_save_path = os.path.join("SyntheticData", "TVAE", folder, task, "synthetic_data.csv")
                    balanced_synthetic_data_save_path = os.path.join("BalancedSyntheticData", "TVAE", folder, task, "synthetic_data.csv")

                    # Create the directories if they don't exist
                    os.makedirs(os.path.dirname(syn_save_path), exist_ok=True)
                    os.makedirs(os.path.dirname(balanced_synthetic_data_save_path), exist_ok=True)

                    syn.to_csv(syn_save_path, index=False)
                    balanced_synthetic_data.to_csv(balanced_synthetic_data_save_path, index=False)
                    
                    print("Generated Synthetic data for {} and stored in path sucessfully".format(folder))  

                else:
                    print(f"'additional_info.json' file not found in '{task_directory}'")

            else:
                print(f"'training_data.csv' not found in '{task_directory}'")

        else:
            print(f"'semi-supervised' directory not found in '{folder}'")
            
            
     
            
        
        
