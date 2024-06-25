
from scipy.spatial import distance
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import random
import matplotlib.pyplot as plt
random.seed(42)


def concatenate_and_scale(df1, df2, df3, continuous_features):
    """
    Concatenate three dataframes row-wise and perform standard scaling on all columns.

    Parameters:
    - df1, df2, df3: Dataframes to be concatenated.

    Returns:
    - Scaled dataframes for each input dataframe.
    """
    # Concatenate dataframes row-wise
    concatenated_df = pd.concat([df1, df2, df3], axis=0)

    # Perform standard scaling on all columns
    scaler = StandardScaler()
    if continuous_features is not None:
        scaled_data = scaler.fit_transform(concatenated_df[continuous_features])
        concatenated_df[continuous_features] = scaled_data

    # Determine the number of rows in each original dataframe
    len_df1, len_df2, len_df3 = len(df1), len(df2), len(df3)

    # Create new dataframes with scaled data
    scaled_df1 = pd.DataFrame(concatenated_df.iloc[:len_df1, :], columns=concatenated_df.columns)
    scaled_df2 = pd.DataFrame(concatenated_df.iloc[len_df1:len_df1+len_df2, :], columns=concatenated_df.columns)
    scaled_df3 = pd.DataFrame(concatenated_df.iloc[len_df1+len_df2:, :], columns=concatenated_df.columns)

    return scaled_df1, scaled_df2, scaled_df3

def canberra_modified(a,b):
    return np.sqrt(np.sum(np.array(
        [np.abs(1.0 - x) / (1.0 + np.abs(x)) for x in (np.abs(a-b) + 1.0)]
        )))
def _canberra_modified(a, b):
    distance=np.zeros((int(a.shape[0]), int(b.shape[0])))
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            distance[i,j]=canberra_modified(a[i], b[j])
    return distance


class MembershipInferenceAttack:
    def __init__(self, threshold=0.4, access=0.1):
        """
        Initialize the MembershipInferenceAttack object with specified threshold and access fraction.
        """
        self.threshold = threshold
        self.access = access

    def access_to_attack(self, data):
        """
        Randomly sample a fraction of the data for the attack.
        """
        return data.sample(frac=self.access, random_state=64)

    def construct_real(self, training_data, holdout_data, need_label=True):
        """
        Combine training and holdout data, scale it, and add a column indicating training data.
        """
        
        if need_label:
            training_data = training_data.copy()
            holdout_data = holdout_data.copy()
            
            training_data['is_training'] = 1
            holdout_data['is_training'] = 0
            
        real_data = pd.concat([training_data, holdout_data], ignore_index=True)
        
        return real_data

    def calculate_distance(self, access_to_attack, synthetic_data):
        """
        Calculate Hamming distances between access_to_attack data and synthetic data.
        """
        distances = distance.cdist(np.array(access_to_attack), np.array(synthetic_data), 'hamming')
        #distances = _canberra_modified(np.array(access_to_attack), np.array(synthetic_data))
        return distances

    def distance_to_predictions(self, distances):
        """
        Convert distances to binary predictions based on the threshold.
        """
        flattened_array = distances.min(axis=1)
        predictions = (flattened_array <= self.threshold).astype(int)
        return predictions

    def calculate_metrics(self, true_labels, predicted_labels):
        """
        Calculate accuracy and precision based on ground truth and predictions.
        """
        accuracy = np.mean(true_labels == predicted_labels)

        true_positives = np.sum((true_labels == 1) & (predicted_labels == 1))
        false_positives = np.sum((true_labels == 0) & (predicted_labels == 1))

        precision = true_positives / max((true_positives + false_positives), 1)

        return accuracy, precision

    def attack(self, training_data, holdout_data, synthetic_data):
        """
        Perform the membership inference attack.
        """
        real_data = self.construct_real(training_data, holdout_data)
        access_to_attack = self.access_to_attack(real_data)
        distances = self.calculate_distance(access_to_attack.drop(['is_training'], axis=1), synthetic_data)
        predictions = self.distance_to_predictions(distances)
        ground_truth = np.array(access_to_attack.iloc[:, -1:]).flatten()
        accuracy, precision = self.calculate_metrics(true_labels=ground_truth, predicted_labels=predictions)
        return accuracy, precision

    
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor



    
    
class AttributeInferenceAttack:
    def __init__(self, training_data, holdout_data, synthetic_data, QID, risk_features=None, continuous_features=None, access=0.5):
        self.training_data = training_data
        self.holdout_data = holdout_data
        self.synthetic_data = synthetic_data
        self.QID = QID
        self.risk_features=risk_features
        self.access = access
        self.continuous_features = continuous_features
        self.column_names = list(synthetic_data.columns)

    def access_to_attack(self, data):
        return data.sample(frac=self.access, random_state=64)

    def construct_real(self, training_data, holdout_data, need_label=True):
        real_data = pd.concat([training_data, holdout_data], ignore_index=True)
        return real_data

    def predict_categorical(self, synthetic_data, access_to_attack, feature):
        features_access_to_attack = access_to_attack[self.QID]
        clf = DecisionTreeClassifier(random_state=64)  # Use DecisionTreeClassifier 
        X = synthetic_data[self.QID]
        y = synthetic_data[feature]
        clf.fit(X, y)
        predicted_labels = clf.predict(features_access_to_attack)
        accuracy = accuracy_score(access_to_attack[feature], predicted_labels)
        f1 = f1_score(access_to_attack[feature], predicted_labels, average='weighted')
        return accuracy, f1

    def predict_numeric(self, synthetic_data, access_to_attack, feature):
        features_access_to_attack = access_to_attack[self.QID]
        reg = DecisionTreeRegressor(random_state=64)  # Use DecisionTreeRegressor 
        X = synthetic_data[self.QID]
        y = synthetic_data[feature]

        reg.fit(X, y)
        predicted_values = reg.predict(features_access_to_attack)
        rmse = np.sqrt(mean_squared_error(access_to_attack[feature], predicted_values))
        return rmse

    def attribute_inference_attack(self):
        self.training_data.columns = self.column_names
        self.holdout_data.columns = self.column_names
        if self.continuous_features is None:
            self.continuous_features = []
        categorical_features = [x for x in self.column_names if x not in self.continuous_features]
        real_data = self.construct_real(self.training_data, self.holdout_data, need_label=False)
        real_data.columns = self.column_names
        access_to_attack = self.access_to_attack(data=real_data)

        accuracies = []
        f1_scores = []
        rmses = []
        
        if self.risk_features is not None:
            for name in self.risk_features:
                if name not in self.QID:
                    if name in categorical_features:
                        accuracy, f1 = self.predict_categorical(self.synthetic_data, access_to_attack, name)
                        accuracies.append(accuracy)
                        f1_scores.append(f1)

                    elif name in self.continuous_features:
                        rmse = self.predict_numeric(self.synthetic_data, access_to_attack, name)
                        rmses.append(rmse)
        else:
            for name in self.column_names:
                if name not in self.QID:
                    if name in categorical_features:
                        accuracy, f1 = self.predict_categorical(self.synthetic_data, access_to_attack, name)
                        accuracies.append(accuracy)
                        f1_scores.append(f1)

                    elif name in self.continuous_features:
                        rmse = self.predict_numeric(self.synthetic_data, access_to_attack, name)
                        rmses.append(rmse)

        return np.mean(accuracies), np.mean(f1_scores), np.mean(rmses)
    
    
    
def MIA(training_data, holdout_data, synthetic_data, threshold=[0.4, 0.3, 0.2, 0.1], access=[0.2, 0.4, 0.6, 0.8, 1]):
    Pre=np.zeros([len(access),len(threshold)], dtype=float)
    Acc=np.zeros([len(access),len(threshold)], dtype=float)
    for i in range(len(access)):
        for j in range(len(threshold)):
            attack_instance=MembershipInferenceAttack(threshold[j], access[i])
            accuracy, precision= attack_instance.attack(training_data, holdout_data, synthetic_data)
            Pre[i,j]=precision
            Acc[i,j]=accuracy
    # Create structured arrays for accuracy and precision
    dtype = [('Threshold', float), ('Access', float), ('Accuracy', float), ('Precision', float)]
    combined_structured_array = np.zeros((len(access) * len(threshold),), dtype=dtype)

    # Populate the structured array
    for i, a in enumerate(access):
        for j, t in enumerate(threshold):
            idx = i * len(threshold) + j
            combined_structured_array[idx]['Threshold'] = t
            combined_structured_array[idx]['Access'] = a
            combined_structured_array[idx]['Accuracy'] = Acc[i, j]
            combined_structured_array[idx]['Precision'] = Pre[i, j]
            
    return combined_structured_array




def AIA(training_data, holdout_data, synthetic_dataframe_list, QID, risk_features, continuous_features, model_names):
    accuracies=[]
    f1_scores=[]
    rmse_scores=[]
    for synthetic_data in synthetic_dataframe_list:
        train_data, hold_data, syn_data = concatenate_and_scale(training_data, holdout_data, synthetic_data, continuous_features)
        attack_instance= AttributeInferenceAttack(train_data, hold_data, syn_data, QID, risk_features, continuous_features)
        accuracy, f1_score, rmse = attack_instance.attribute_inference_attack()
        accuracies.append(accuracy)
        f1_scores.append(f1_score)
        rmse_scores.append(rmse)
    report={"Model":model_names, "Accuracy": accuracies, "F1_score": f1_scores, "RMSE":rmse_scores}
    df=pd.DataFrame(report)
    return df




def plot_and_save(structured_array, folder="Avg.", model_name=None, file_path=None, plot_name="plot"):
    plt.figure(figsize=(8, 8))

    thresholds = np.unique(structured_array['Threshold'])
    colors = ['grey', 'green', 'blue', 'red']

    for i, t in enumerate(thresholds):
        accuracy_subset = structured_array[structured_array['Threshold'] == t]['Accuracy']
        precision_subset = structured_array[structured_array['Threshold'] == t]['Precision']
        access_subset = structured_array[structured_array['Threshold'] == t]['Access']

        # Round off the threshold value to 1 decimal point for legend
        rounded_threshold = round(t, 1)

        plt.plot(access_subset, precision_subset, label=f'Threshold={rounded_threshold}', linestyle='-', marker='o', color=colors[i])

    plt.xlabel('Access')
    plt.ylabel('Precision')
    #plt.title('Precision for each Threshold: ' + str(model_name) + " " + str(folder))
    plt.legend()

    plt.xlim(0, 1.2)
    plt.ylim(0, 1.2)

    if file_path is not None:
        # Save the plot with high resolution to the specified file path and name
        plt.savefig(f'{file_path}/{plot_name}.png', dpi=300)
    else:
        plt.savefig(f'{plot_name}.png', dpi=300)

        
def add_combined_array(combined_structured_array_0,combined_structured_array_1):
    combined_structured_array2 = np.zeros_like(combined_structured_array_0, dtype=[('Threshold', float), ('Access', float), ('Accuracy', float), ('Precision', float)])
    for i in range(len(combined_structured_array_0)):
        for j in range(len(combined_structured_array_0[0])):
            combined_structured_array2[i][j]=combined_structured_array_0[i][j] + combined_structured_array_1[i][j] 
    return combined_structured_array2

def avg_combined_array(combined_structured_array_0, N_datasets):
    combined_structured_array1 = np.zeros_like(combined_structured_array_0, dtype=[('Threshold', float), ('Access', float), ('Accuracy', float), ('Precision', float)])
    for i in range(len(combined_structured_array_0)):
        for j in range(len(combined_structured_array_0[0])):
            combined_structured_array1[i][j]=combined_structured_array_0[i][j] / N_datasets 
    return combined_structured_array1  