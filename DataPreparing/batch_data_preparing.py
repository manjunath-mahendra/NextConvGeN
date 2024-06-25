import os
import warnings
warnings.filterwarnings("ignore")
import json
import numpy as np
from prepare_data import *

# Remember: Here we use test_size as 0.5 for privacy evaluation otherwise 0.3
test_size = 0.3
privacy_test_size = 0.5



def process_data_new(csv_file, task_name, output_dir_NextConvGen, output_dir_tabddpm, test_size=0.3, seed=None):
    csv_path = os.path.join(dir_PreprocessedDatasets, csv_file)
    json_file = csv_file.split('.')[0].replace('SS', '') + '.json'
    json_path = os.path.join(dir_PreprocessedDatasets, json_file)
    random_state = seed
    
    if not os.path.isfile(json_path):
        return
    
    # Call the function split_data from prepare_data.py
    
    categorical_features, numerical_features, target, training_data, holdout_data, training_data_NextConvGeN, additional_info = split_data_NextConvGeN(csv_path, json_path,  test_size=test_size, random_state=seed, task=task_name)
   
    
    df1 = training_data
    df2 = holdout_data
    df3 = training_data_NextConvGeN
    
    if df1 is not None and df2 is not None and df3 is not None:
        # Create a new directory with folder name as dataset name to store the preprocessed datasets
        dataset_name = csv_file.split(".")[0].replace('SS', '')
        dataset_dir = os.path.join(output_dir_NextConvGen, dataset_name, task_name)
        os.makedirs(dataset_dir, exist_ok=True)

        # Assign the file names
        json_filename = f'additional_info.json'
        training_filename = f'training_data.csv'
        holdout_filename = f'holdout_data.csv'
        next_convgen_filename = f'NextConvGeN_training_data.csv'

        # Save the csv files in directory PreparedDatasets 
        df1.to_csv(os.path.join(dataset_dir, training_filename), index=False)
        df2.to_csv(os.path.join(dataset_dir, holdout_filename), index=False)
        df3.to_csv(os.path.join(dataset_dir, next_convgen_filename), index=False)

        additional_info_ = {
            'ordered_features': additional_info['ordered_features'],
            'target': additional_info['target'],
            'indices_continuous_features': additional_info.get('indices_continuous_features', None),
            'indices_ordinal_features': additional_info.get('indices_ordinal_features', None),
            'indices_nominal_features': additional_info.get('indices_nominal_features', None),
            'target_value_counts': {int(key[0]) if isinstance(key, tuple) else int(key) if isinstance(key, str) else key: value
                                    for key, value in additional_info['target_value_counts'].items()}
        }
        with open(os.path.join(dataset_dir, json_filename), "w") as json_file:
            json.dump(additional_info_, json_file)

        print(f"Prepared {dataset_name} data and saved successfully")


    # Now for TabDDPM:
    dataset_name = csv_file.split(".")[0].replace('SS', '')
    dataset_dir = os.path.join(output_dir_tabddpm, dataset_name, task_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    X_num_train, X_num_test, X_num_val, X_cat_train, X_cat_test, X_cat_val, y_train, y_test, y_val = split_data_tabddpm(categorical_features, numerical_features, target, training_data, random_state=seed)
    
    info = {
        'name': dataset_name,
        'id': dataset_name,
        'task_type': "multiclass",
        'n_num_features': X_num_train.shape[1],
        'n_cat_features': X_cat_test.shape[1],
        'train_size': X_num_train.shape[0],
        'val_size': X_num_val.shape[0],
        'test_size': X_cat_test.shape[0]
    }

    np.save(os.path.join(dataset_dir, 'X_num_train.npy'), X_num_train)
    np.save(os.path.join(dataset_dir, 'X_num_test.npy'), X_num_test)
    np.save(os.path.join(dataset_dir, 'X_num_val.npy'), X_num_val)
    np.save(os.path.join(dataset_dir, 'X_cat_train.npy'), X_cat_train)
    np.save(os.path.join(dataset_dir, 'X_cat_test.npy'), X_cat_test)
    np.save(os.path.join(dataset_dir, 'X_cat_val.npy'), X_cat_val)
    np.save(os.path.join(dataset_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(dataset_dir, 'y_test.npy'), y_test)
    np.save(os.path.join(dataset_dir, 'y_val.npy'), y_val)

    with open(os.path.join(dataset_dir, 'info.json'), "w") as json_file:
        json.dump(info, json_file)

    print(f"Prepared {dataset_name} data in TabDDPM format and saved successfully")



current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
dir_PreprocessedDatasets = os.path.join(parent_directory, 'PreprocessedDatasets')
dir_PreparedData = os.path.join(parent_directory, 'PreparedData1')
dir_TabDDPM_datasets = os.path.join(parent_directory, 'TabDDPM_datasets1')
dir_privacy_PreparedData = os.path.join(parent_directory, 'PreparedDataPrivacy1')
dir_privacy_TabDDPM_datasets = os.path.join(parent_directory, 'TabDDPM_privacy_datasets1')

# List all CSV files in the directory
csv_files = [f for f in os.listdir(dir_PreprocessedDatasets) if f.endswith('.csv')]



# Process each CSV file for supervised and semi-supervised tasks
for csv_file in csv_files:
    if not csv_file.startswith('SS'):
        process_data_new(csv_file, "supervised", dir_PreparedData, dir_TabDDPM_datasets, test_size, seed=11)
        process_data_new(csv_file, "supervised", dir_privacy_PreparedData, dir_privacy_TabDDPM_datasets, privacy_test_size, seed=11)
    else:
        process_data_new(csv_file, "semi-supervised", dir_PreparedData, dir_TabDDPM_datasets, test_size, seed=11)
        process_data_new(csv_file, "semi-supervised", dir_privacy_PreparedData, dir_privacy_TabDDPM_datasets, privacy_test_size, seed=11)



        
        
# Print a message to indicate the process is complete
print("Data split and saved to directories.")



    

