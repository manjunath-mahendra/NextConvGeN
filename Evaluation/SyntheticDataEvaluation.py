import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression
from scipy.spatial import distance
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import statistics
from scipy import stats
import scipy
import torch
from torchmetrics.functional.nominal import theils_u as torch_theils_u
import math
from math import sqrt
from scipy.stats import chi2_contingency, entropy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer, f1_score
import random
import umap
from sklearn.cluster import KMeans
random.seed(42)

def avg_absolute_difference(X, Y):
    return np.average(np.absolute(X-Y))

def get_categorical_correlations(df):
    """
    Calculate Theil's U coefficients for all pairs of categorical columns in a DataFrame.

    Parameters:
    - df: DataFrame containing the categorical columns

    Returns:
    - flattened_list: 2D array containing Theil's U coefficients for all pairs of columns
    """
    # Identify columns with numeric values and convert them to strings
    numeric_columns = df.select_dtypes(include=['int', 'float']).columns
    df[numeric_columns] = df[numeric_columns].astype(str)

    # Convert all remaining non-string columns to object type
    non_string_columns = df.select_dtypes(exclude=['object']).columns
    df[non_string_columns] = df[non_string_columns].astype(str)

    categorical_columns = df.columns

    num_columns = len(categorical_columns)
    theils_u_array = []

    for i in range(num_columns):
        theils_u_row = []
        for j in range(i + 1, num_columns):
            col1 = categorical_columns[i]
            col2 = categorical_columns[j]
            
            # Calculate Theil's U using torchmetrics library
            preds = torch.tensor(df[col1].astype('category').cat.codes)
            target = torch.tensor(df[col2].astype('category').cat.codes)
            theils_u_value = torch_theils_u(preds, target)

            theils_u_row.append(theils_u_value.item())

        theils_u_array.append(theils_u_row)
        flattened_list = [value for sublist in theils_u_array for value in sublist]

    return np.array(flattened_list)


def FeatureCorrelation(real_data, SyntheticDataFrameList, continuous_features, nominal_features, ordinal_features):
    categorical_features = nominal_features + ordinal_features

    # Calculating correlations for continuous features
    continuous_correlation_diff = []
    real_corr_matrix_continuous = real_data[continuous_features].corr()
    for syn_data in SyntheticDataFrameList:
        synthetic_corr_matrix_continuous = syn_data[continuous_features].corr()
        diff_from_baseline = avg_absolute_difference(real_corr_matrix_continuous, synthetic_corr_matrix_continuous)
        continuous_correlation_diff.append(diff_from_baseline)

    # Calculating correlations for categorical features
    categorical_correlation_diff = []
    real_corr_matrix_categorical = get_categorical_correlations(real_data[categorical_features])
    for syn_data in SyntheticDataFrameList:
        synthetic_corr_matrix_categorical = get_categorical_correlations(syn_data[categorical_features])
        diff_from_baseline = avg_absolute_difference(real_corr_matrix_categorical, synthetic_corr_matrix_categorical)
        categorical_correlation_diff.append(diff_from_baseline)

    avg_correlation_diff = [statistics.mean(k) for k in zip(continuous_correlation_diff, categorical_correlation_diff)]

    return avg_correlation_diff


def calculate_propensity_scores(real_data, synthetic_data, label_column, random_state=42):
    """
    Calculate propensity scores for a dataset using a RandomForestClassifier.

    Parameters:
    - real_data (pandas DataFrame): The DataFrame containing real data.
    - synthetic_data (pandas DataFrame): The DataFrame containing synthetic data.
    - label_column (str): The name of the column that contains class labels (1 for real, 0 for synthetic).
    - random_state (int): Random seed for reproducibility.

    Returns:
    - propensity_scores (pandas Series): The propensity scores for each data point in the combined dataset.
    """
    # Add labels to the real and synthetic data
    real_data[label_column] = 0
    synthetic_data[label_column] = 1

    # Combine the real and synthetic data
    combined_data = pd.concat([real_data, synthetic_data], ignore_index=True)

    # Split the combined data into features and labels
    features = combined_data.drop(columns=[label_column])
    labels = combined_data[label_column]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=random_state)

    # Train a RandomForestClassifier
    classifier = LogisticRegression(random_state=42)
    classifier.fit(X_train, y_train)

    # Calculate propensity scores
    probability = pd.DataFrame(classifier.predict_proba(features))
    moving_sum=0
    for i in probability[1]:
        moving_sum+=np.square(i - 0.5)
    propensity_score= moving_sum / probability.shape[0]
    real_data.drop([label_column], axis=1, inplace=True)
    synthetic_data.drop([label_column], axis=1, inplace=True)
        
    return propensity_score



def PredictReal(real_data, synthetic_data, label_column, random_state=42):
    """
    Predict the class labels for real data using Logistic Regression and return the average F1 score
    from 10-fold cross-validation.

    Parameters:
    - real_data (pandas DataFrame): The DataFrame containing real data.
    - synthetic_data (pandas DataFrame): The DataFrame containing synthetic data.
    - label_column (str): The name of the column that contains class labels (1 for real, 0 for synthetic).
    - random_state (int): Random seed for reproducibility.

    Returns:
    - avg_f1_score (float): The average F1 score from 10-fold cross-validation.
    """
    # Add labels to the real and synthetic data
    real_data[label_column] = 1
    synthetic_data[label_column] = 0

    # Combine the real and synthetic data
    combined_data = pd.concat([real_data, synthetic_data], ignore_index=True)

    # Split the combined data into features and labels
    features = combined_data.drop(columns=[label_column])
    labels = combined_data[label_column]

    # Train a Logistic Regression model
    classifier = LogisticRegression(random_state=random_state)

    # Calculate F1 score using cross-validation
    f1_scorer = make_scorer(f1_score)
    cross_val_scores = cross_val_score(classifier, features, labels, cv=10, scoring=f1_scorer)

    # Calculate the average F1 score
    avg_f1_score = np.mean(cross_val_scores)

    # Drop the temporary label column
    real_data.drop([label_column], axis=1, inplace=True)
    synthetic_data.drop([label_column], axis=1, inplace=True)

    return avg_f1_score


def student_t_tests(real, synthetic, cont_feature_list) :
    p_values = []
    #loop to perform the tests for each attribute
    for c in cont_feature_list :
        _, p = stats.ttest_ind(np.array(real[c]), np.array(synthetic[c]))
        p_values.append(p)

    #return the obtained p-values
    return np.mean(p_values)



def kl_divergence(p, q):
    """Calculate KL divergence between two categorical distributions."""
    return entropy(p, q)

def kl_divergence_average(real, synthetic, ord_list, nom_list):
    """Calculate average KL divergence feature-wise, excluding features with infinite KL divergence."""
    # Get list of categorical column names
    cat_features = ord_list + nom_list
    
    kl_divergences = []

    for feature in cat_features:
        # Calculate value counts normalized for both real and synthetic datasets
        p_counts = real[feature].value_counts(normalize=True, sort=False)
        q_counts = synthetic[feature].value_counts(normalize=True, sort=False)

        # Ensure that both p and q have the same set of possible outcomes
        all_values = set(p_counts.index) | set(q_counts.index)
        p = p_counts.reindex(all_values, fill_value=0)
        q = q_counts.reindex(all_values, fill_value=0)

        # Calculate KL divergence for the current feature
        kl_div = kl_divergence(p, q)
        
        # Check if KL divergence is finite
        if np.isfinite(kl_div):
            kl_divergences.append(kl_div)

    # Return the mean KL divergence across all finite KL divergences
    if kl_divergences:
        return np.mean(kl_divergences)
    else:
        return np.nan  # Return NaN if no finite KL divergences were found





def cluster_wise_df(dataframe, cluster_label_list):
    cluster_df_list = []
    for cluster_label in np.unique(cluster_label_list):
        cluster_df = dataframe[dataframe['ClusterLabel'] == cluster_label].copy()
        cluster_df_list.append(cluster_df)

    return cluster_df_list


def normalize(data_frame):
    scaler = StandardScaler()
    columns=data_frame.columns
    scaled_features = scaler.fit_transform(data_frame)
    data_frame=pd.DataFrame(scaled_features, columns=columns)
    return data_frame
    

def K_means_clustering(real_data,synthetic_data,n_cluster=2, on_low_dim=True):
    real_data['IsReal']=1
    synthetic_data['IsReal']=0
    combined_data=pd.concat([real_data, synthetic_data], axis=0)
    Isreal=combined_data['IsReal']
    scaled_data=normalize(combined_data.drop(['IsReal'], axis=1))
    if on_low_dim :
        reducer=umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        low_emb=reducer.fit_transform(scaled_data)
        kmeans = KMeans(n_clusters=n_cluster, random_state=42).fit(low_emb)
    else:
        kmeans = KMeans(n_clusters=n_cluster,random_state=42 ).fit(scaled_data)
    combined_data['ClusterLabel']=kmeans.labels_
    real_data.drop(['IsReal'], axis=1, inplace=True)
    synthetic_data.drop(['IsReal'], axis=1, inplace=True)
    return combined_data   

def LogClusterMetric(LabeledDataFrame):
    ProportionOfReal=LabeledDataFrame[LabeledDataFrame['IsReal']==1].shape[0] / LabeledDataFrame.shape[0]
    cluster_df_list=cluster_wise_df(LabeledDataFrame,LabeledDataFrame.ClusterLabel.to_list())
    ProportionOfRealInClusters=0
    for i in range(len(cluster_df_list)):
        ProportionOfRealInClusters+=((cluster_df_list[i][cluster_df_list[i].IsReal==1].shape[0] / cluster_df_list[i].shape[0]) - ProportionOfReal)**2
    if ProportionOfRealInClusters==0:
        ProportionOfRealInClusters+=10**-9
    LCG = math.log(ProportionOfRealInClusters / len(cluster_df_list))
    return LCG
        
        
def cluster_wise_val_score(ref_list,pred_list):
    def safeDiv(a, b):
        if b != 0:
            return a / b
        return 0.0
    
    F1_score_list = []
    Geometric_mean_list = []
    cluster_score_list = []
    true_positive_total = 0
    for i in np.unique(ref_list):
        indices = [j for j,val in enumerate(ref_list) if val == i]
        true_positive = 0
        for index in indices:
            if i == pred_list[index]:
                true_positive += 1
        true_positive_total += true_positive
        
        precision = safeDiv(true_positive, pred_list.count(i))
        recall = safeDiv(true_positive, len(indices))
        F1_score = safeDiv(2.0 * precision * recall, precision + recall)
        GM = np.sqrt(precision * recall)
        cluster_score = recall * 100.0
        
        #print("F1_Score of cluster "+str(i)+" is {}".format(F1_score))
        #print("Geometric mean of cluster "+str(i)+" is {}".format(GM))
        #print("Correctly predicted data points in cluster "+str(i)+" is {}%".format(cluster_score))
        #print("\n")
        F1_score_list.append((ref_list.count(i)/len(ref_list))*F1_score)
        Geometric_mean_list.append((ref_list.count(i)/len(ref_list))*GM)
        cluster_score_list.append((ref_list.count(i)/len(ref_list))*cluster_score)

    #correctly_predicted = safeDiv(100.0 * true_positive_total, len(ref_list))

    #print("weigted average F1_Score of all clusters is {}".format(np.sum(F1_score_list)))
    #print("weighted average Geometric mean of all clusters is {}".format(np.sum(Geometric_mean_list)))
    #print("weighted average of Correctly predicted data points in all clusters is {}%".format(np.sum(cluster_score_list)))
    return np.sum(F1_score_list), np.sum(Geometric_mean_list)

def HoldOutAnalysis(real_train_data, hold_out_data, synthetic_df_list, target ='Target', algorithim_names=['NextConvGeN', 'TabDDPM']):
    clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=42)
    cv_score_real=np.average(cross_val_score(clf, real_train_data.drop([target],axis=1), real_train_data[target],cv=5))
    #print(cv_score_real)
    on_real=clf.fit(normalize(real_train_data.drop([target],axis=1)), real_train_data[target])
    predicted_holdout=on_real.predict(hold_out_data.drop([target],axis=1))
    F1_real, GM_real = cluster_wise_val_score(list(hold_out_data[target]),list(predicted_holdout))
    cv_diff=[]
    F1_score_diff=[]
    GM_diff=[]
    for synth_data in synthetic_df_list:
        cv_score_synth=np.average(cross_val_score(clf, synth_data.drop([target],axis=1), synth_data[target],cv=5))
        #print(cv_score_synth)
        cv_diff.append(abs(cv_score_real - cv_score_synth))
        on_synth=clf.fit(normalize(synth_data.drop([target],axis=1)), synth_data[target])
        predicted_holdout=on_synth.predict(hold_out_data.drop([target],axis=1))
        F1_synth, GM_synth = cluster_wise_val_score(list(hold_out_data[target]), list(predicted_holdout))
        F1_score_diff.append(abs(F1_real-F1_synth))
        GM_diff.append(abs(GM_real-GM_synth))
    return cv_diff, F1_score_diff, GM_diff


def calculate_feature_importance_difference(real_data, synthetic_df_list,hold_out, target_column='Target'):
    # Create an empty list to store the feature importance
    feature_importance_differences_scores=[]
    
    X_train_real = real_data.drop([target_column],axis=1)
    y_train_real = real_data[target_column]
    
    X_test_real = hold_out.drop([target_column],axis=1)
    y_test_real = hold_out[target_column]
    
    
    
    for synthetic_data in synthetic_df_list:
        X_train_synth = synthetic_data.drop([target_column],axis=1)
        y_train_synth = synthetic_data[target_column]
        
        feature_importance_differences = []


        # Iterate over each feature
        for feature in real_data.columns:
            if feature != target_column:
                # Train Gradient Boosting Classifier on real data without the feature
                real_classifier_without_feature = DecisionTreeClassifier(random_state=42)
                real_classifier_without_feature.fit(X_train_real.drop([feature],axis=1), y_train_real)


                # Train Gradient Boosting Classifier on synthetic data without the feature
                synth_classifier_without_feature = DecisionTreeClassifier(random_state=42)
                synth_classifier_without_feature.fit(X_train_synth.drop([feature],axis=1), y_train_synth)

                # Predict on real data and synthetic data with and without the feature
                F1_without_feature_real, _ = cluster_wise_val_score(list(y_test_real),list(real_classifier_without_feature.predict(X_test_real.drop([feature],axis=1))))
                F1_without_feature_synth, _ = cluster_wise_val_score(list(y_test_real),list(synth_classifier_without_feature.predict(X_test_real.drop([feature],axis=1))))

                # Calculate the absolute difference in predictions
                _diff = abs(F1_without_feature_real - F1_without_feature_synth)

                # Append the difference to the list
                feature_importance_differences.append(_diff)

        # Calculate the average of the differences
        average_difference = sum(feature_importance_differences) / len(feature_importance_differences)
        feature_importance_differences_scores.append(average_difference)
    

    return feature_importance_differences_scores

"""
def calculate_feature_importance_difference(real_data, synthetic_df_list, target_column='Target'):
    # Create an empty list to store the feature importance
    feature_importance_differences_scores=[]
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(real_data.drop([target_column],axis=1), real_data[target_column], test_size=0.30, random_state=42)
    real_classifier_with_feature = GradientBoostingClassifier(random_state=42)
    real_classifier_with_feature.fit(X_train_real, y_train_real)
    _, GM_with_feature_real= cluster_wise_val_score(list(y_test_real),list(real_classifier_with_feature.predict(X_test_real)))
    
    for synthetic_data in synthetic_df_list:
        X_train_synth, X_test_synth, y_train_synth, y_test_synth = train_test_split(synthetic_data.drop([target_column],axis=1), synthetic_data[target_column], test_size=0.30, random_state=42)
        synth_classifier_with_feature = GradientBoostingClassifier(random_state=42)
        synth_classifier_with_feature.fit(X_train_synth, y_train_synth)
        _, GM_with_feature_synth= cluster_wise_val_score(list(y_test_synth),list(synth_classifier_with_feature.predict(X_test_synth)))
        feature_importance_differences = []


        # Iterate over each feature
        for feature in real_data.columns:
            if feature != target_column:
                # Train Gradient Boosting Classifier on real data without the feature
                real_classifier_without_feature = GradientBoostingClassifier(random_state=42)
                real_classifier_without_feature.fit(X_train_real.drop([feature],axis=1), y_train_real)


                # Train Gradient Boosting Classifier on synthetic data without the feature
                synth_classifier_without_feature = GradientBoostingClassifier(random_state=42)
                synth_classifier_without_feature.fit(X_train_synth.drop([feature],axis=1), y_train_synth)

                # Predict on real data and synthetic data with and without the feature
                _, GM_without_feature_real= cluster_wise_val_score(list(y_test_real),list(real_classifier_without_feature.predict(X_test_real.drop([feature],axis=1))))
                _, GM_without_feature_synth= cluster_wise_val_score(list(y_test_synth),list(synth_classifier_without_feature.predict(X_test_synth.drop([feature],axis=1))))

                # Calculate the absolute difference in predictions
                real_diff = abs(GM_with_feature_real - GM_without_feature_real)
                synthetic_diff = abs(GM_with_feature_synth - GM_without_feature_synth)

                # Calculate the absolute difference between real and synthetic data
                difference = abs(real_diff - synthetic_diff)

                # Append the difference to the list
                feature_importance_differences.append(difference)

        # Calculate the average of the differences
        average_difference = sum(feature_importance_differences) / len(feature_importance_differences)
        feature_importance_differences_scores.append(average_difference)
    

    return feature_importance_differences_scores
"""
def scale_data(df) :
    """Scale a dataframe to get the values between 0 and 1. It returns the scaled dataframe.
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        The dataframe to scale

    Returns
    -------
    pandas.core.frame.DataFrame
        A dataframe with the scaled data
    """

    #initialize and fit the scaler
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)

    #return the scaled dataframe
    return pd.DataFrame(scaled)


def similarity_evaluation(synthetic_data, real_data) :
    """Compute the pairwise euclidean distances between each pair of real and synthetic records
    
    Parameters
    ----------
    synthetic_data : numpy.ndarray
        Synthetic data records
    real_data : numpy.ndarray
        Real data records

    """

    #compute the pairwise euclidean distances
    distances = distance.cdist(synthetic_data, real_data, 'euclidean')
    
    #compute the hausdorff distance
    HausdorffDist = scipy.spatial.distance.directed_hausdorff(synthetic_data, real_data)[0]
    
    #compute the cosine similarity between each pair of synthetic and real records
    CosineDist = cosine_similarity(synthetic_data, real_data)
    
    #the mean and std values of the computed pairwise euclidean distances
    EucDist_mean = np.round(np.mean(distances),4)
    EucDist_std = np.round(np.std(distances),4)
    
    #the computed value rounded on 4 decimals
    HausdorffDist= np.round(HausdorffDist,4)
    
    #the mean value of the computed similarity values
    CosineDist_mean = np.round(np.mean(CosineDist),4)

    
    return EucDist_mean, EucDist_std, HausdorffDist, CosineDist_mean


def SyntheticdataEvaluationReport(real_data, hold_out, SyntheticDataFrameList, ContinuousFeatures, OrdinalFeatures, NominalFeatures, target_column='Target', ModelNameList=['NextConvGeN', 'TabDDPM'] ):
    # Univariate analaysis
    StudentTTest=[]
    KL_div=[]
    PropensityScore=[]
    PredictRealOrSyn=[]
    LogMetric=[]
    EuclideanDistance_Mean=[]
    EuclideanDistance_Std=[]
    HausdorffDistance=[]
    CosineSimilarity=[]
    
    for synthetic_data in SyntheticDataFrameList:
        t_test_p_val=student_t_tests(real_data, synthetic_data, ContinuousFeatures)
        StudentTTest.append(t_test_p_val)
        KL_diver=kl_divergence_average(real_data, synthetic_data, OrdinalFeatures, NominalFeatures)
        KL_div.append(KL_diver)
        propensity_score=calculate_propensity_scores(real_data, synthetic_data, label_column="IsReal")
        PropensityScore.append(propensity_score)
        PredictReal_=PredictReal(real_data, synthetic_data, label_column="IsReal", random_state=42)
        PredictRealOrSyn.append(PredictReal_)
        LCM=LogClusterMetric(K_means_clustering(real_data,synthetic_data, 2, False))
        LogMetric.append(LCM)
        EucDist_mean, EucDist_std, HausdorffDist, CosineDist_mean=similarity_evaluation(np.array(scale_data(synthetic_data)), np.array(scale_data(real_data)))
        EuclideanDistance_Mean.append(EucDist_mean)
        EuclideanDistance_Std.append(EucDist_std)
        HausdorffDistance.append(HausdorffDist)
        CosineSimilarity.append(CosineDist_mean)
    FeatureCorrelations=FeatureCorrelation(real_data, SyntheticDataFrameList, ContinuousFeatures, NominalFeatures, OrdinalFeatures)
    CrossValidation, F1Score, GeometricMean= HoldOutAnalysis(real_data, hold_out, SyntheticDataFrameList, target = target_column)
    FeatureConsistency=calculate_feature_importance_difference(real_data, SyntheticDataFrameList, hold_out, target_column=target_column)
    import pandas as pd

    data = {
        'Model': ModelNameList,
        'Student-T-test': StudentTTest,
        'KL divergence': KL_div,
        'PropensityScore': PropensityScore,
        'PredictRealOrSyn':PredictRealOrSyn,
        'Log Cluster Metric': LogMetric,
        'Feature Correlations Difference': FeatureCorrelations,
        'Cross-validation': CrossValidation,
        'F-1 score': F1Score,
        'Geometric Mean': GeometricMean,
        'Feature-consistency': FeatureConsistency,
        'EuclideanDistance_Mean': EuclideanDistance_Mean,
        'EuclideanDistance_Std': EuclideanDistance_Std,
        'HausdorffDistance': HausdorffDistance,
        'CosineSimilarity': CosineSimilarity
    }

    Report = pd.DataFrame(data)
    #Report.set_index('Model', inplace=True)
    #print(Report)
    
    return Report

        