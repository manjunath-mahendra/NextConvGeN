import math
import numpy as np
from scipy.spatial import distance
import fdc.tools as tools


def create_total_impute(data, distance_matrix, missing_value_list):
    def create_value_list(f, index):
        index_in_dist_mat = np.where(distance_matrix[:,0] == index)[0][0]
        value_list = []
        for neb_index in distance_matrix[index_in_dist_mat][1:]:
            impute_value = data.loc[[neb_index]][f]
            if float(impute_value) == float(impute_value):
                value_list.append(float(impute_value))
            
            if len(value_list) >= 6:
                break

        return np.array(value_list)

    def feature_impute_master(f):
        missing_value_indices = data[data[f].isnull()].index.tolist()
        return np.array([
            create_value_list(f, index)
            for index in missing_value_indices
            ])

    def imputed_value(row):
        intcounter = tools.count(lambda x: (not math.isnan(x)) and 0 == (x - int(x)), row)

        if intcounter == len(row):
            return np.array(np.bincount(row.astype(int)).argmax())
        else:
            return np.array(np.mean(row))

    total_impute_master = [
        feature_impute_master(f)
        for f in missing_value_list
        ]

    return [
        [imputed_value(row) for row in plane]
        for plane in total_impute_master
        ]


def create_distance_matrix_old(dense_data):
    dense_data_index = np.array(dense_data.index)
    dense_data = np.array(dense_data)

    return np.array([
        dense_data_index[
            np.argsort([ distance.euclidean(x, y) for y in dense_data ])
            ]
        for x in dense_data
        ])


def create_distance_matrix(dense_data):
    dense_data_index = np.array(dense_data.index)
    dense_data = np.array(dense_data)
    size = len(dense_data)

    matrix = [[ None for i in range(size)] for j in range(size)]

    # Calculate the squared euclidian distances.
    for nx, x in enumerate(dense_data):
        b = dense_data - x
        matrix[nx] = np.sum(b*b, axis=1)

    # Calculate the indices and replace the distance rows.
    # So we create our result matrix and do cleanup at the same time.
    for n in range(size):
        matrix[n] = dense_data_index[ np.argsort(matrix[n]) ]

    return np.array(matrix)




def fix_missing_values(data, limit=4):
    dense_data_pool = list(data.isna().sum().index[data.isna().sum() < limit])

    dense_data = data[dense_data_pool].dropna()

    data = data.loc[np.array(dense_data.index)]
    
    distance_matrix = create_distance_matrix(dense_data)

    missing_value_list = [ x
        for x in list(data.columns)
        if x not in dense_data_pool
        ]

    total_impute = create_total_impute(
        data, distance_matrix, missing_value_list)

    for f, value in enumerate(missing_value_list):
        missing_value_indices = data[data[value].isnull()].index.tolist()
        for i, value_index in enumerate(missing_value_indices):
            data.at[value_index, value] = total_impute[f][i]

    return data


