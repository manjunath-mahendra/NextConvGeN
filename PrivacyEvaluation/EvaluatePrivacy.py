import pandas as pd
import numpy as np
import os
import json
import random
from privacy import *
random.seed(42)

def EvaluatePrivacy(tn, task = "semi-supervised"):
    path = os.getcwd()
    parent_path = os.path.abspath(os.path.join(path, os.pardir))
    base_directory = os.path.join(parent_path, "PreparedDataPrivacy"+tn[1])
    print(base_directory)
    syn_directory = os.path.join(parent_path, tn + "_BalancedSyntheticDataPrivacy")
    ResultDataFrameList = []
    DatasetNames=[]
    count=0
    with open("DatasetInfo.json", 'r') as info_file:
        feature_info = json.load(info_file)
        #print(feature_info)
        #feature_info = json.load(info_file)
    
    # List all subdirectories (folders) in the base directory
    all_folders = [folder for folder in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, folder))]

    # Iterate through all the folders
    for folder in all_folders:
        SyntheticDataFrameList = []
        info_path = os.path.join(base_directory, folder)
        nextconvgen_path = os.path.join(syn_directory,'NextConvGeN', folder)
        ctgan_path = os.path.join(syn_directory,'CTGAN', folder)
        convexctgan_path = os.path.join(syn_directory,'convexCTGAN', folder)
        ctabgan_path = os.path.join(syn_directory,'CTABGAN', folder)
        convexctabgan_path = os.path.join(syn_directory,'convexCTABGAN', folder)
        tabddpm_path = os.path.join(syn_directory,'TabDDPM', folder)

        # Check if the directory exists in the current folder
        info_directory = os.path.join(info_path, task)
        real_data_directory = os.path.join(info_directory, "training_data.csv")
        holdout_data_directory = os.path.join(info_directory, "holdout_data.csv")
        nextconvgen_directory = os.path.join(nextconvgen_path, task, "synthetic_data.csv")
        ctgan_directory = os.path.join(ctgan_path, task, "synthetic_data.csv")
        convexctgan_directory = os.path.join(convexctgan_path, task, "synthetic_data.csv")
        ctabgan_directory = os.path.join(ctabgan_path, task, "synthetic_data.csv")
        convexctabgan_directory = os.path.join(convexctabgan_path, task, "synthetic_data.csv")
        tabddpm_directory = os.path.join(tabddpm_path, task, "synthetic_data.csv")
        folder_feature_info = feature_info[folder]
        if os.path.exists(info_directory):

            # Load the data
            real_data = pd.read_csv( real_data_directory)
            holdout_data = pd.read_csv( holdout_data_directory)
            nextconvgen_data = pd.read_csv(nextconvgen_directory)
            SyntheticDataFrameList.append(nextconvgen_data)
            ctgan_data = pd.read_csv(ctgan_directory)
            SyntheticDataFrameList.append(ctgan_data)
            convexctgan_data = pd.read_csv(convexctgan_directory)
            SyntheticDataFrameList.append(convexctgan_data)
            ctabgan_data = pd.read_csv(ctabgan_directory)
            SyntheticDataFrameList.append(ctabgan_data)
            convexctabgan_data = pd.read_csv(convexctabgan_directory)
            SyntheticDataFrameList.append(convexctabgan_data)
            tabddpm_data = pd.read_csv(tabddpm_directory)
            SyntheticDataFrameList.append(tabddpm_data)
            #print(info_directory)

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
            
            
            nextconvgen_combined_structured_array = MIA(real_data, holdout_data, nextconvgen_data)
            #print(nextconvgen_combined_structured_array)
            if count==0:
                nextconvgen_sum = ctgan_sum = convexctgan_sum = ctabgan_sum = convexctabgan_sum = tabddpm_sum = np.zeros_like(nextconvgen_combined_structured_array, dtype=[('Threshold', float), ('Access', float), ('Accuracy', float), ('Precision', float)])
            nextconvgen_sum=add_combined_array(nextconvgen_sum,nextconvgen_combined_structured_array)
            
            
            ctgan_combined_structured_array = MIA(real_data, holdout_data, ctgan_data)
            ctgan_sum=add_combined_array(ctgan_sum,ctgan_combined_structured_array)
            
            convexctgan_combined_structured_array = MIA(real_data, holdout_data, convexctgan_data)
            convexctgan_sum=add_combined_array(convexctgan_sum,convexctgan_combined_structured_array)
            
            
            ctabgan_combined_structured_array = MIA(real_data, holdout_data, ctabgan_data)
            ctabgan_sum=add_combined_array(ctabgan_sum,ctabgan_combined_structured_array)
            
            convexctabgan_combined_structured_array = MIA(real_data, holdout_data, convexctabgan_data)
            convexctabgan_sum=add_combined_array(convexctabgan_sum,convexctabgan_combined_structured_array)
            
            
            tabddpm_combined_structured_array = MIA(real_data, holdout_data, tabddpm_data)
            #print(tabddpm_combined_structured_array)
            tabddpm_sum=add_combined_array(tabddpm_sum,tabddpm_combined_structured_array)
            #print(tabddpm_sum)
            
            
            #MIA_reports.append(MIA_report)
            

            report_save_path = os.path.join(tn + "_PrivacyEvaluationReport", folder, task)
            # Create path for NextConvGeN
            nextconvgen_mia_path = os.path.join(report_save_path, "NextConvGeN_MIA_scores")
            # Create the directories if they don't exist
            os.makedirs(os.path.dirname(nextconvgen_mia_path), exist_ok=True)
            np.save(nextconvgen_mia_path, nextconvgen_combined_structured_array)
            plot_and_save(nextconvgen_combined_structured_array,folder, "NextConvGeN", report_save_path, plot_name="NextConvGeN")
            
            # Create path for CTGAN
            ctgan_mia_path = os.path.join(report_save_path, "CTGAN_MIA_scores")
            # Create the directories if they don't exist
            os.makedirs(os.path.dirname(ctgan_mia_path), exist_ok=True)
            np.save(ctgan_mia_path, ctgan_combined_structured_array)
            plot_and_save(ctgan_combined_structured_array,folder,"CTGAN", report_save_path, plot_name = "CTGAN")
            
            # Create path for convexCTGAN
            convexctgan_mia_path = os.path.join(report_save_path, "convexCTGAN_MIA_scores")
            # Create the directories if they don't exist
            os.makedirs(os.path.dirname(convexctgan_mia_path), exist_ok=True)
            np.save(convexctgan_mia_path, convexctgan_combined_structured_array)
            plot_and_save(convexctgan_combined_structured_array,folder,"convexCTGAN", report_save_path, plot_name = "convexCTGAN") 
            
            # Create path for CTABGAN
            ctabgan_mia_path = os.path.join(report_save_path, "CTABGAN_MIA_scores")
            # Create the directories if they don't exist
            os.makedirs(os.path.dirname(ctabgan_mia_path), exist_ok=True)
            np.save(ctabgan_mia_path, ctabgan_combined_structured_array)
            plot_and_save(ctabgan_combined_structured_array,folder, "CTABGAN", report_save_path, plot_name="CTABGAN")
            
            # Create path for convexCTABGAN
            convexctabgan_mia_path = os.path.join(report_save_path, "convexCTABGAN_MIA_scores")
            # Create the directories if they don't exist
            os.makedirs(os.path.dirname(convexctabgan_mia_path), exist_ok=True)
            np.save(convexctabgan_mia_path, convexctabgan_combined_structured_array)
            plot_and_save(convexctabgan_combined_structured_array,folder, "convexCTABGAN", report_save_path, plot_name="convexCTABGAN")
            
            # Create path for TabDDPM
            tabddpm_mia_path = os.path.join(report_save_path, "TabDDPM_MIA_scores")
            # Create the directories if they don't exist
            os.makedirs(os.path.dirname(tabddpm_mia_path), exist_ok=True)
            np.save(tabddpm_mia_path, tabddpm_combined_structured_array)
            plot_and_save(tabddpm_combined_structured_array,folder, "TabDDPM", report_save_path, plot_name="TabDDPM")
            
            count+=1
            
            evaluation_report = AIA(training_data=real_data, holdout_data=holdout_data, synthetic_dataframe_list = SyntheticDataFrameList, QID  = folder_feature_info["quasi_identifiers"], risk_features = None, continuous_features=continuous_list, model_names=["NextConvGeN", "CTGAN","convexCTGAN", "CTABGAN", "convexCTABGAN", "TabDDPM"])
            aia_report_save_path = os.path.join(report_save_path, "AIA_report.csv")
            evaluation_report.to_csv(aia_report_save_path, index=True)

            #evaluation_report=evaluation_report.set_index('Model')

            ResultDataFrameList.append(evaluation_report)
            DatasetNames.append(folder)

            #print("dataframe list: ", ResultDataFrameList)

            print("Evaluated Synthetic data for {} and stored the report in path sucessfully".format(folder))

        else:
            print(f"Directory not found in '{folder}'")
    if task=='supervised':
        n=10
    else:
        n=12
    #print(nextconvgen_sum)    
    nextconvgen_sum = avg_combined_array(nextconvgen_sum,n)
    #print(nextconvgen_sum)
    np_path = os.path.join(tn, "avg_nextconvgen")
    np.save(np_path, nextconvgen_sum)
    plot_and_save(nextconvgen_sum,"Avg.","NextConvGeN",file_path=tn, plot_name = task + "_avg_nextconvgen")
            
    
    ctgan_sum = avg_combined_array(ctgan_sum, n)
    np.save("avg_ctgan", ctgan_sum)
    plot_and_save(ctgan_sum,"Avg.","CTGAN", file_path=tn, plot_name = task + "_avg_ctgan")
    
    convexctgan_sum = avg_combined_array(convexctgan_sum, n)
    np.save("avg_convexctgan", convexctgan_sum)
    plot_and_save(convexctgan_sum,"Avg.","convexCTGAN", file_path=tn, plot_name = task + "_avg_convexctgan")
    
    ctabgan_sum = avg_combined_array(ctabgan_sum, n)
    np.save("avg_CTABGAN", ctabgan_sum)
    plot_and_save(ctabgan_sum,"Avg.","CTABGAN", file_path=tn, plot_name = task + "_avg_CTABGAN")
    
    convexctabgan_sum = avg_combined_array(convexctabgan_sum, n)
    np.save("avg_convexCTABGAN", convexctabgan_sum)
    plot_and_save(convexctabgan_sum,"Avg.","convexCTABGAN", file_path=tn, plot_name = task + "_avg_convexCTABGAN")
    
    tabddpm_sum = avg_combined_array(tabddpm_sum, n)
    np.save("avg_tabddpm", tabddpm_sum)
    plot_and_save(tabddpm_sum,"Avg.","TabDDPM", file_path=tn, plot_name = task + "_avg_tabddpm")
    
    # Create a new dataframe by combining the dataframes and adding a 'Dataset_Name' column
    new_dataframe_list = [pd.concat([pd.DataFrame([name]*df.shape[0], columns=['Dataset Name']), df], axis=1) for name, df in zip(DatasetNames, ResultDataFrameList)]

    # Concatenate all the dataframes in the new list along rows
    result_dataframe = pd.concat(new_dataframe_list, ignore_index=False)
    
    combined_report_path = os.path.join(tn, task+"Combined_report.csv")
    # Create the directories if they don't exist
    os.makedirs(os.path.dirname(combined_report_path), exist_ok=True)

    result_dataframe.to_csv(combined_report_path)

    # Use the `concat` method to stack the dataframes on top of each other
    #stacked_df = pd.concat(ResultDataFrameList)

    #print("Stacked DataFrame is: ",stacked_df)

    # Use the `groupby` method to group by the index (rows) and calculate the mean
    #average_df = stacked_df.groupby(level=0).mean()

    # If you want to reset the index to the default integer index
    #average_df = average_df.reset_index()

    #average_df.to_csv(task+'AverageEvaluationReport.csv')


EvaluatePrivacy("t1", task = "semi-supervised")
EvaluatePrivacy("t1", task = "supervised")

EvaluatePrivacy("t2", task = "semi-supervised")
EvaluatePrivacy("t2", task = "supervised")

EvaluatePrivacy("t3", task = "semi-supervised")
EvaluatePrivacy("t3", task = "supervised")

EvaluatePrivacy("t4", task = "semi-supervised")
EvaluatePrivacy("t4", task = "supervised")

EvaluatePrivacy("t5", task = "semi-supervised")
EvaluatePrivacy("t5", task = "supervised")