# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:32:38 2022

@author: Hattie
"""

import pandas as pd
import numpy as np
import scipy.io

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def extract_emg_features(emg, repetition, step, N):
    """
    Function to split emg data into windows and extract feature for each
    channel in the window

    Parameters
    ----------
    emg : numpy.ndarray
        An array of emg data.
    repetition : numpy.ndarray
        An array of labels for each row of emg data.
    step : int
        The size of the step between windows.
    N : int
        The size of the window.

    Returns
    -------
    window_features : list
        A list of lists of features for each emg window.

    """

    # Create a list to store features for each window
    window_features = []
    
    # Iterate over the emg data in windows of 100
    for i in range(0, len(emg), step):
        # Get the current window of emg and repetition data
        temp_emg = emg[i:i+N]
        temp_repetition = repetition[i:i+N]
        # If there is a crossover between labels skip feature extraction
        if len(np.unique(temp_repetition)) > 1:
            continue
        # Create a list to store the features for each channel
        features = []
        # Iterate over each channel of the current window
        for j in range(0, temp_emg.shape[1]):
            # Append the mean and standard deviation of each channel to the 
            # list of features
            features.append(np.mean(temp_emg[:,j]))
            features.append(np.median(temp_emg[:,j]))
            features.append(np.std(temp_emg[:,j]))
            features.append(np.var(temp_emg[:,j]))
            features.append(np.min(temp_emg[:,j]))
            features.append(np.max(temp_emg[:,j]))
            
        # Append the unique repetition values to the list of features
        features.append(int(np.unique(temp_repetition)))
        # Append the list of features for one window to the list for all windows
        window_features.append(features)
        
    return window_features



if __name__ == "__main__":
    print("Hello world")
    
    # Load a .mat file 
    mat = scipy.io.loadmat('data/S1_E1_A1.mat')
    
    # Extract the emg data and repetition data
    extracted_features = extract_emg_features(mat['emg'], mat['repetition'], 100, 500)
     
    # Create dataframe of emg features
    df = pd.DataFrame(extracted_features)
    
    # Standardise the features
    X = StandardScaler().fit_transform(df.iloc[:,:-1].values)
    # Encode the labels
    y = LabelEncoder().fit_transform(df.iloc[:,-1].values)
    
    # Split data into train and test sets using 70/30
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train and predict logisitic regression
    clf = LogisticRegression(random_state=0, max_iter=10000).fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("Logistic Regression")
    print(classification_report(y_test,predictions))
    
    # Train and predict a SVM
    clf = SVC(gamma='auto').fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("SVM")
    print(classification_report(y_test,predictions))
    
    # Train and predict a random forest
    clf = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("Random Forest")
    print(classification_report(y_test,predictions))