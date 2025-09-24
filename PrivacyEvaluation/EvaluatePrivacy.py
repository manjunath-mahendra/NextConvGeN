import pandas as pd
import numpy as np
import os
import json
import random
from privacy import *
random.seed(42)

def EvaluatePrivacy(tn, task="semi-supervised"):
    path = os.getcwd()
    parent_path = os.path.abspath(os.path.join(path, os.pardir))
    base_directory = os.path.join(parent_path, f"PreparedDataPrivacy{tn[-1]}")
    syn_directory = os.path.join(parent_path, tn + "_BalancedSyntheticDataPrivacy")
    ResultDataFrameList = []
    DatasetNames = []
    count = 0
    with open("DatasetInfo.json", 'r') as info_file:
        feature_info = json.load(info_file)

    all_folders = [folder for folder in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, folder))]

    for folder in all_folders:
        print(f"dataset:{folder}")
        SyntheticDataFrameList = []
        info_path = os.path.join(base_directory, folder)
        print(f"path:{info_path}")

        # Model directories
        nextconvgen_path = os.path.join(syn_directory, 'NextConvGeN', folder)
        ctgan_path       = os.path.join(syn_directory, 'CTGAN', folder)
        ctabgan_path     = os.path.join(syn_directory, 'CTABGAN', folder)
        tabddpm_path     = os.path.join(syn_directory, 'TabDDPM', folder)
        cart_path        = os.path.join(syn_directory, 'CART', folder)
        datasynth_path   = os.path.join(syn_directory, 'DataSynth', folder)
        great_path       = os.path.join(syn_directory, 'GReaT', folder)

        info_directory = os.path.join(info_path, task)
        real_data_directory = os.path.join(info_directory, "training_data.csv")
        holdout_data_directory = os.path.join(info_directory, "holdout_data.csv")

        # Synthetic data paths
        nextconvgen_directory = os.path.join(nextconvgen_path, task, "synthetic_data.csv")
        ctgan_directory       = os.path.join(ctgan_path, task, "synthetic_data.csv")
        ctabgan_directory     = os.path.join(ctabgan_path, task, "synthetic_data.csv")
        tabddpm_directory     = os.path.join(tabddpm_path, task, "synthetic_data.csv")
        cart_directory        = os.path.join(cart_path, task, "synthetic_data.csv")
        datasynth_directory   = os.path.join(datasynth_path, task, "synthetic_data.csv")
        great_directory       = os.path.join(great_path, task, "synthetic_data.csv")

        folder_feature_info = feature_info[folder]

        if os.path.exists(info_directory):
            # Load the data
            real_data = pd.read_csv(real_data_directory)
            holdout_data = pd.read_csv(holdout_data_directory)

            # Load synthetic data
            nextconvgen_data = pd.read_csv(nextconvgen_directory); SyntheticDataFrameList.append(nextconvgen_data)
            ctgan_data       = pd.read_csv(ctgan_directory);       SyntheticDataFrameList.append(ctgan_data)
            ctabgan_data     = pd.read_csv(ctabgan_directory);     SyntheticDataFrameList.append(ctabgan_data)
            tabddpm_data     = pd.read_csv(tabddpm_directory);     SyntheticDataFrameList.append(tabddpm_data)
            cart_data        = pd.read_csv(cart_directory);        SyntheticDataFrameList.append(cart_data)
            datasynth_data   = pd.read_csv(datasynth_directory);   SyntheticDataFrameList.append(datasynth_data)
            great_data       = pd.read_csv(great_directory);       SyntheticDataFrameList.append(great_data)

            # Feature info
            information_path = os.path.join(info_directory, "additional_info.json")
            with open(information_path, 'r') as info_file:
                info = json.load(info_file)

            ordinal_list, nominal_list, continuous_list = [], [], []
            if info['indices_ordinal_features'] is not None:
                ordinal_list.extend([info['ordered_features'][i] for i in info['indices_ordinal_features']])
            if info['indices_nominal_features'] is not None:
                nominal_list.extend([info['ordered_features'][i] for i in info['indices_nominal_features']])
            if info['indices_continuous_features'] is not None:
                continuous_list.extend([info['ordered_features'][i] for i in info['indices_continuous_features']])

            if isinstance(info.get("target"), str):
                target = info['target']
            else:
                target = info['target'][0]
            if info['target'] is not None:
                nominal_list.append(target)

            # --- MIA for each model ---
            model_names = ["NextConvGeN", "CTGAN", "CTABGAN", "TabDDPM", "CART", "DataSynth", "GReaT"]
            synthetic_dict = {
                "NextConvGeN": nextconvgen_data,
                "CTGAN": ctgan_data,
                "CTABGAN": ctabgan_data,
                "TabDDPM": tabddpm_data,
                "CART": cart_data,
                "DataSynth": datasynth_data,
                "GReaT": great_data
            }

            if count == 0:
                # initialize sums
                mia_sums = {name: np.zeros_like(
                    MIA(real_data, holdout_data, synthetic_dict[name]),
                    dtype=[('Threshold', float), ('Access', float), ('Accuracy', float), ('Precision', float)]
                ) for name in model_names}

            # run and save results per dataset
            report_save_path = os.path.join(tn + "_PrivacyEvaluationReport", folder, task)
            os.makedirs(report_save_path, exist_ok=True)

            for name, syn_data in synthetic_dict.items():
                combined_structured_array = MIA(real_data, holdout_data, syn_data)
                mia_sums[name] = add_combined_array(mia_sums[name], combined_structured_array)

                np.save(os.path.join(report_save_path, f"{name}_MIA_scores"), combined_structured_array)
                plot_and_save(combined_structured_array, folder, name, report_save_path, plot_name=name)

            count += 1

            # --- AIA ---
            evaluation_report = AIA(
                training_data=real_data,
                holdout_data=holdout_data,
                synthetic_dataframe_list=list(synthetic_dict.values()),
                QID=folder_feature_info["quasi_identifiers"],
                risk_features=None,
                continuous_features=continuous_list,
                model_names=model_names
            )
            aia_report_save_path = os.path.join(report_save_path, "AIA_report.csv")
            evaluation_report.to_csv(aia_report_save_path, index=True)

            ResultDataFrameList.append(evaluation_report)
            DatasetNames.append(folder)
            print(f"Evaluated Synthetic data for {folder} and stored the report successfully")
        else:
            print(f"Directory not found in '{folder}'")

    # --- Average results ---
    n = 10 if task == 'supervised' else 12
    os.makedirs(tn, exist_ok=True)

    for name, arr_sum in mia_sums.items():
        avg_array = avg_combined_array(arr_sum, n)
        np.save(os.path.join(tn, f"avg_{name.lower()}"), avg_array)
        plot_and_save(avg_array, "Avg.", name, file_path=tn, plot_name=f"{task}_avg_{name.lower()}")

    # --- Combine all AIA reports ---
    new_dataframe_list = [pd.concat([pd.DataFrame([name]*df.shape[0], columns=['Dataset Name']), df], axis=1)
                          for name, df in zip(DatasetNames, ResultDataFrameList)]
    result_dataframe = pd.concat(new_dataframe_list, ignore_index=False)
    combined_report_path = os.path.join(tn, "Combined_report.csv")
    os.makedirs(os.path.dirname(combined_report_path), exist_ok=True)
    result_dataframe.to_csv(combined_report_path)


# Run evaluations
#EvaluatePrivacy("t1", task="semi-supervised")
EvaluatePrivacy("t1", task="supervised")

#EvaluatePrivacy("t2", task="semi-supervised")
EvaluatePrivacy("t2", task="supervised")

#EvaluatePrivacy("t3", task="semi-supervised")
EvaluatePrivacy("t3", task="supervised")

#EvaluatePrivacy("t4", task="semi-supervised")
EvaluatePrivacy("t4", task="supervised")

#EvaluatePrivacy("t5", task="semi-supervised")
EvaluatePrivacy("t5", task="supervised")
