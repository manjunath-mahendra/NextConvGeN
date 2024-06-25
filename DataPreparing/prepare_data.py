import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json

    

def NumericToFeatures (synth_data, features):
    """
    Converts numeric column names into their original names
    
    """
    synth_data.columns=features
    return synth_data

def FeaturesToNumeric(training_data, features):
    """
    Replace feature names with numbers
    """
    dicts = {feature: i for i, feature in enumerate(features)}
    training_data.rename(columns=dicts, inplace=True)

def prepare_for_NextConvGeN(data, features):
    data_copy = data.copy() 
    FeaturesToNumeric(data_copy, features)
    return data_copy

def split_data_internal(data, json_file, task='supervised'):
    """
    Split data into training and holdout sets based on the task and data structure.

    Args:
        data (DataFrame): The input data path.
        json_file (str): Path to a JSON file containing feature information.
        task (str, optional): The task type, either 'supervised' or 'semi-supervised'. Default is 'supervised'.\


    """
    
    data=pd.read_csv(data)
    
    # Load feature information from the JSON file
    with open(json_file, 'r') as file:
        feature_info = json.load(file)

    # Extract feature lists
    continuous_features = feature_info['cont_list']
    nominal_features = feature_info['nom_list']
    ordinal_features = feature_info['ord_list']
    # Reorder the data features as continuous + nominal + ordinal
    ordered_features = (continuous_features or []) + (ordinal_features or []) + (nominal_features or []) 

    if task == 'supervised':
        # Split the data into X (features) and y (target)
        target = feature_info['target']
        X = data[ordered_features]
        y = data[target]
    elif task == 'semi-supervised':
        # Set the target to 'Target' for semi-supervised learning
        target = 'Target'
        X = data[ordered_features]
        y = data[target]
    else:
        raise ValueError("Invalid task. Task must be 'supervised' or 'semi-supervised'.")
        
    #ordered_features.append(target[0]) if type(target)==list else ordered_features.append(target)

    return continuous_features, nominal_features, ordinal_features, ordered_features, target, X, y


def split_data_tabddpm(categorical_features, numerical_features, target, training_data, random_state=None):
    """
    Split data into training and holdout sets based on the task and data structure.

    """
    
    features = (numerical_features or []) + (categorical_features or [])
    __, X_test_ddpm, _, y_test_ddpm = train_test_split(training_data[features], training_data[target], test_size=0.3, random_state=random_state)

    X_num_train = np.array(training_data[numerical_features])
    X_num_test = np.array(X_test_ddpm[numerical_features])
    X_num_val = np.array(X_test_ddpm[numerical_features])

    X_cat_train = np.array(training_data[categorical_features])
    X_cat_test = np.array(X_test_ddpm[categorical_features])
    X_cat_val = np.array(X_test_ddpm[categorical_features])

    y_train = np.array(training_data[target])
    y_test = np.array(y_test_ddpm)
    y_val = np.array(y_test_ddpm)


    return X_num_train, X_num_test, X_num_val, X_cat_train, X_cat_test, X_cat_val, y_train, y_test, y_val


def split_data_NextConvGeN(data, json_file, test_size=0.3, random_state=None, task='supervised'):
    """
    Split data into training and holdout sets based on the task and data structure.

    Args:
        data (DataFrame): The input data path.
        json_file (str): Path to a JSON file containing feature information.
        test_size (float, optional): The proportion of the data to include in the holdout set. Default is 0.3.
        random_state (int, optional): The random seed for reproducibility. Default is 42.
        task (str, optional): The task type, either 'supervised' or 'semi-supervised'. Default is 'supervised'.\
        tabddpm_format (Boolean): to prepare data in the format TabDDPM needs. Default is 'False' 

    Returns:
        training_data (DataFrame): The training data.
        holdout_data (DataFrame): The holdout data.
        training_data_NextConvGeN (DataFrame): The training data for NextConvGeN model
    """
    
    continuous_features, nominal_features, ordinal_features, ordered_features, target, X, y = split_data_internal(data, json_file, task)
    
    categorical_features = (ordinal_features or []) + (nominal_features or [])
    numerical_features = (continuous_features or [])
    
    # Perform supervised train-test split
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create training and holdout DataFrames
    training_data = pd.concat([X_train, y_train], axis=1)
    holdout_data = pd.concat([X_holdout, y_holdout], axis=1)
    training_data_NextConvGeN=prepare_for_NextConvGeN(training_data, ordered_features)
    additional_info = {
        'ordered_features': ordered_features,
        'target': target,
        'indices_continuous_features': [training_data.columns.get_loc(feat) for feat in continuous_features] if continuous_features is not None else None,
        'indices_ordinal_features': [training_data.columns.get_loc(feat) for feat in ordinal_features] if ordinal_features is not None else None,
        'indices_nominal_features': [training_data.columns.get_loc(feat) for feat in nominal_features] if nominal_features is not None else None,
        'target_value_counts': training_data[target].value_counts().to_dict()
    }
    return categorical_features, numerical_features, target, training_data, holdout_data, training_data_NextConvGeN, additional_info
