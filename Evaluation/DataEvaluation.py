import pandas as pd
import numpy as np
from SyntheticDataEvaluation import *
import os
import json
import random
random.seed(42)

def EvaluateData(real_data_repo=None, task = "semi-supervised", tn=None):
    path = os.getcwd()
    parent_path = os.path.abspath(os.path.join(path, os.pardir))
    base_directory = os.path.join(parent_path, real_data_repo)
    syn_directory = os.path.join(parent_path, tn + "_BalancedSyntheticData")
    ResultDataFrameList = []
    DatasetNames=[]
    # List all subdirectories (folders) in the base directory
    all_folders = [folder for folder in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, folder))]

    # Iterate through all the folders
    for folder in all_folders:
        print(folder)
        SyntheticDataFrameList = []
        info_path = os.path.join(base_directory, folder)
        nextconvgen_path = os.path.join(syn_directory,'NextConvGeN', folder)
        
        ctgan_path = os.path.join(syn_directory,'CTGAN', folder)
        #convexctgan_path = os.path.join(syn_directory,'convexCTGAN', folder)

        ctabgan_path = os.path.join(syn_directory,'CTABGAN', folder)
        #convexctabgan_path = os.path.join(syn_directory,'convexCTABGAN', folder)

        CART_path = os.path.join(syn_directory,'CART', folder)

        DataSynth_path = os.path.join(syn_directory,'DataSynth', folder)

        GReaT_path = os.path.join(syn_directory,'GReaT', folder)

        #Tabula_path = os.path.join(syn_directory,'Tabula', folder)

        tabddpm_path = os.path.join(syn_directory,'TabDDPM', folder)

        # Check if the directory exists in the current folder
        info_directory = os.path.join(info_path, task)
        real_data_directory = os.path.join(info_directory, "training_data.csv")
        holdout_data_directory = os.path.join(info_directory, "holdout_data.csv")
        nextconvgen_directory = os.path.join(nextconvgen_path, task, "synthetic_data.csv")

        ctgan_directory = os.path.join(ctgan_path, task, "synthetic_data.csv")
        #convexctgan_directory = os.path.join(convexctgan_path, task, "synthetic_data.csv")

        ctabgan_directory = os.path.join(ctabgan_path, task, "synthetic_data.csv")
        #convexctabgan_directory = os.path.join(convexctabgan_path, task, "synthetic_data.csv")
        CART_directory = os.path.join(CART_path, task, "synthetic_data.csv")

        GReaT_directory = os.path.join(GReaT_path, task, "synthetic_data.csv")

        DataSynth_directory = os.path.join(DataSynth_path, task, "synthetic_data.csv")

        #Tabula_directory = os.path.join(Tabula_path, task, "synthetic_data.csv")

        tabddpm_directory = os.path.join(tabddpm_path, task, "synthetic_data.csv")


        if os.path.exists(info_directory):

            # Load the data
            real_data = pd.read_csv( real_data_directory)
            holdout_data = pd.read_csv( holdout_data_directory)
            nextconvgen_data = pd.read_csv(nextconvgen_directory)
            SyntheticDataFrameList.append(nextconvgen_data)

            ctgan_data = pd.read_csv(ctgan_directory)
            SyntheticDataFrameList.append(ctgan_data)

            #convexctgan_data = pd.read_csv(convexctgan_directory)
            #SyntheticDataFrameList.append(convexctgan_data)

            ctabgan_data = pd.read_csv(ctabgan_directory)
            SyntheticDataFrameList.append(ctabgan_data)

            #convexctabgan_data = pd.read_csv(convexctabgan_directory)
            #SyntheticDataFrameList.append(convexctabgan_data)

            CART_data = pd.read_csv(CART_directory)
            SyntheticDataFrameList.append(CART_data)

            GReaT_data = pd.read_csv(GReaT_directory)
            SyntheticDataFrameList.append(GReaT_data)

            DataSynth_data = pd.read_csv(DataSynth_directory)
            SyntheticDataFrameList.append(DataSynth_data)

            #Tabula_data = pd.read_csv(Tabula_directory)
            #SyntheticDataFrameList.append(Tabula_data)

            tabddpm_data = pd.read_csv(tabddpm_directory)
            SyntheticDataFrameList.append(tabddpm_data)
            print(info_directory)

            # Load additional info from the 'additional_info.json' file in the 'semi-supervised' directory
            information_path = os.path.join(info_directory, "additional_info.json")
            with open(information_path, 'r') as info_file:
                info = json.load(info_file)

            # Extract information from the info dictionary
            ordinal_list = []
            nominal_list = []
            continuous_list = []

            if info['indices_ordinal_features'] is not None:
                ordinal_list.extend([info['ordered_features'][i] for i in info['indices_ordinal_features']])

            if info['indices_nominal_features'] is not None:
                nominal_list.extend([info['ordered_features'][i] for i in info['indices_nominal_features']])

            if info['indices_continuous_features'] is not None:
                continuous_list.extend([info['ordered_features'][i] for i in info['indices_continuous_features']])

            if isinstance(info.get("target"), str):
                target=info['target']
            else:
                target=info['target'][0]

            # Add the target column
            if info['target'] is not None:
                nominal_list.append(target)

            evaluation_report = SyntheticdataEvaluationReport(real_data,holdout_data, SyntheticDataFrameList, continuous_list, ordinal_list, nominal_list, target_column=target, ModelNameList=['NextConvGeN', 'CTGAN', 'CTABGAN', 'CART','DataSynth','GReaT', 'TabDDPM'])

            # Create path
            result_save_path = os.path.join(tn + "_EvaluationReport", folder, task, "evaluation_report.csv")

            # Create the directories if they don't exist
            os.makedirs(os.path.dirname(result_save_path), exist_ok=True)

            evaluation_report.to_csv(result_save_path, index=True)

            #evaluation_report=evaluation_report.set_index('Model')

            ResultDataFrameList.append(evaluation_report)
            DatasetNames.append(folder)

            #print("dataframe list: ", ResultDataFrameList)

            print("Evaluated Synthetic data for {} and stored the report in path sucessfully".format(folder))

        else:
            print(f"Directory not found in '{folder}'")
        
    # Create a new dataframe by combining the dataframes and adding a 'Dataset_Name' column
    new_dataframe_list = [pd.concat([pd.DataFrame([name]*df.shape[0], columns=['Dataset Name']), df], axis=1) for name, df in zip(DatasetNames, ResultDataFrameList)]

    # Concatenate all the dataframes in the new list along rows
    result_dataframe = pd.concat(new_dataframe_list, ignore_index=True)
    
    combined_save_path = os.path.join(tn, task+'Combined_report.csv')

    result_dataframe.to_csv(combined_save_path)

    # Use the `concat` method to stack the dataframes on top of each other
    stacked_df = pd.concat(ResultDataFrameList)

    #print("Stacked DataFrame is: ",stacked_df)

    # Use the `groupby` method to group by the index (rows) and calculate the mean
    average_df = stacked_df.groupby(level=0).mean(numeric_only=True)

    # If you want to reset the index to the default integer index
    average_df = average_df.reset_index()

    #average_df.to_csv(task+'AverageEvaluationReport.csv')


#EvaluateData(real_data_repo="PreparedData1", task = "semi-supervised",tn = "t1")
EvaluateData(real_data_repo="PreparedData1", task = "supervised", tn = "t1")

#EvaluateData(real_data_repo="PreparedData2", task = "semi-supervised",tn = "t2")
EvaluateData(real_data_repo="PreparedData2", task = "supervised", tn = "t2")

#EvaluateData(real_data_repo="PreparedData3", task = "semi-supervised",tn = "t3")
EvaluateData(real_data_repo="PreparedData3", task = "supervised", tn = "t3")

#EvaluateData(real_data_repo="PreparedData4", task = "semi-supervised",tn = "t4")
EvaluateData(real_data_repo="PreparedData4", task = "supervised", tn = "t4")

#EvaluateData(real_data_repo="PreparedData5", task = "semi-supervised",tn = "t5")
EvaluateData(real_data_repo="PreparedData5", task = "supervised", tn = "t5")