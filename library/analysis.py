from library.dataset import DataSet, TrainTestData

import pickle
import numpy as np
import time
import random
import csv
import gzip
import sys
import os
from imblearn.datasets import fetch_datasets


def loadDataset(datasetName):
    def isSame(xs, ys):
        for (x, y) in zip(xs, ys):
            if x != y:
                return False
        return True
    
    def isIn(ys):
        def f(x):
            for y in ys:
                if isSame(x,y):
                    return True
            return False
        return f

    print(f"Load '{datasetName}'")
    if datasetName.startswith("imblearn_"):
        print("from imblearn")
        ds = fetch_datasets()
        myData = ds[datasetName[9:]]
        ds = None

        features = myData["data"]
        labels = myData["target"]
    elif datasetName.startswith("kaggle_"):
        features = []
        labels = []
        c = csv.reader(gzip.open(f"data_input/{datasetName}.csv.gz", "rt")) 
        for (n, row) in enumerate(c):
            # Skip heading
            if n > 0:
                features.append([float(x) for x in row[:-1]])
                labels.append(int(row[-1]))

        features = np.array(features)
        labels = np.array(labels)

    else:
        print("from pickle file")
        pickle_in = open(f"data_input/{datasetName}.pickle", "rb")
        pickle_dict = pickle.load(pickle_in)

        myData = pickle_dict["folding"]
        k = myData[0]

        labels = np.concatenate((k[1], k[3]), axis=0).astype(float)
        features = np.concatenate((k[0], k[2]), axis=0).astype(float)

    label_1 = list(np.where(labels == 1)[0])
    label_0 = list(np.where(labels != 1)[0])
    features_1 = features[label_1]
    features_0 = features[label_0]
    cut = np.array(list(filter(isIn(features_1), features_0)))
    if len(cut) > 0:
        print(f"non empty cut in {datasetName}! ({len(cut)} points)")
    
    ds = DataSet(data0=features_0, data1=features_1)
    print("Data loaded.")
    return ds



def showTime(t):
    s = int(t)
    m = s // 60
    h = m // 60
    d = h // 24
    s = s % 60
    m = m % 60
    h = h % 24
    if d > 0:
        return f"{d} days {h:02d}:{m:02d}:{s:02d}"
    else:
        return f"{h:02d}:{m:02d}:{s:02d}"




    
testSets = [
    "folding_abalone_17_vs_7_8_9_10",
    "folding_abalone9-18",
    "folding_car_good",
    "folding_car-vgood",
    "folding_flare-F",
    "folding_hypothyroid",
    "folding_kddcup-guess_passwd_vs_satan",
    "folding_kr-vs-k-three_vs_eleven",
    "folding_kr-vs-k-zero-one_vs_draw",
    "folding_shuttle-2_vs_5",
    "folding_winequality-red-4",
    "folding_yeast4",
    "folding_yeast5",
    "folding_yeast6",
    #"imblearn_webpage",
    #"imblearn_mammography",
    #"imblearn_protein_homo",
    #"imblearn_ozone_level",
    #"kaggle_creditcard"
    ]