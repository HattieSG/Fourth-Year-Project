# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:32:38 2022

@author: Hattie
"""

import os
import scipy.io
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from imblearn.under_sampling import NearMiss, RandomUnderSampler, EditedNearestNeighbours
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, label_binarize
from sklearn.svm import SVC
from tensorflow.keras import layers, models


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



def increment_labels(stimulus, increment):
    """Function to increment the value of none 0 labels in the emg data
    
        :param stimulus: The list of data labels
        :type stimulus: numpy.ndarray
        
        :param increment: The value to increase each label by
        :type incrememnt: int
        
        :return: The modified labels
        :rtype: numpy.ndarray
    """
    # Convert the array to a list
    rep_list = stimulus.tolist()
    
    # Use list comprehension to add the increment value to each none 0 label
    inc_list = [x[0]+increment if x[0] != 0 else x[0] for x in rep_list]
    
    # Convert the list back to a nx1 array
    inc_array = np.array(inc_list, 'int8').reshape(len(inc_list), 1)
    
    return inc_array



def multiclass_roc(true_labs, pred_b, n_classes, title):
    """Function to plot a ROC_AUC graph for a given set of multi class labels
        and predictions
    
        :param true_labs: The true labels for each prediction
        :type stimulus: numpy.array
        
        :param preb_b: The probability of each class as predicted by a model
        :type incrememnt: numpy.array
        
        :param n_classes: The number of classes in the data
        :type n_classes: int
        
        :param title: The title for the current plot
        :type title: str
        
        :return: Void
        :rtype: None
    """
    
    # Binarize the true labels for the model
    true_b = label_binarize(true_labs, classes=range(0, 13))
    
    # Create dictionaries to store the true postitive rate, false positive rate, and roc_auc
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Iterate over each class
    for i in range(n_classes):
        # Compute ROC curve and ROC area for each class
        fpr[i], tpr[i], _ = roc_curve(true_b[:, i], pred_b[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true_b.ravel(), pred_b.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Ititialise an empty array the same size as all_fpr
    mean_tpr = np.zeros_like(all_fpr)
    # Iterate over each class
    for i in range(n_classes):
        # Then interpolate all ROC curves at this points
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    # Store the macro average for fpr and tpr
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    # Calculate the macro average auc
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot the micro average AUC curve
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]))
    
    # Plot the macro average AUC curve
    plt.plot(fpr["macro"], tpr["macro"],
             label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]))
    
    # Add a straight central line to the plot
    plt.plot([0, 1], [0, 1], "k--")
    # Add a legend to the plot
    plt.legend()
    # Add the title to the plot
    plt.title(title)
    plt.show()



if __name__ == "__main__":
    print("Hello world")
    # List all files in current directory
    files = os.listdir('data/')
    
    extracted_features = []
    start = time.time()
    # Iterate over all of the files
    for file in files:
        print(f"Processing {file}...")
        # If it is a .mat data file read and process it
        if ".mat" in file:
            # Load the .mat file to a dictionary
            mat = scipy.io.loadmat(f"data/{file}")
            print(f"\t{len(mat['stimulus'])} rows in file")
            print(f"\t{len(np.unique(mat['stimulus']))} unique labels in file")
            
            # Split the file by the E tag and modify the labels depending on the value
            e_val = file.split("_")[1]
            
            if e_val == "E1":
                new_stimulus = mat['stimulus']
            elif e_val == "E2":
                continue
                new_stimulus = increment_labels(mat['stimulus'], 12)
            elif e_val == "E3":
                continue
                new_stimulus= increment_labels(mat['stimulus'], 29)
                
            # Extract the emg data and repetition data
            extracted_features.append(extract_emg_features(mat['emg'], new_stimulus, 100, 500))
    
    # Flatten the features for each file into a single list
    all_features = [item for sublist in extracted_features for item in sublist] 
    
    print(f"Feature extraction time: {time.time()-start}")
    
    # Create dataframe of emg features
    df = pd.DataFrame(all_features)
    
    
    # Standardise the features
    start = time.time()
    X = StandardScaler().fit_transform(df.iloc[:,:-1].values)
    print(f"Standardization time: {time.time()-start}")
    
    # # Normalize the features
    # start = time.time()
    # X = MinMaxScaler().fit_transform(df.iloc[:,:-1].values)
    # print(f"Normalization time: {time.time()-start}")
    
    # Encode the labels
    y = LabelEncoder().fit_transform(df.iloc[:,-1].values)
    
    # Create bar plot showing the frequency of the different labels
    xlabs, counts = np.unique(y, return_counts=True)
    bars = plt.bar(xlabs, counts, align='center')
    # Add x labels every 5 bars
    plt.gca().set_xticks(np.arange(0, 53, 5))
    # Add a label to the bar every 5th bar
    ctr = 0
    for bar in bars:
        if ctr % 5 == 0:
            yval = bar.get_height()
            plt.text(bar.get_x(), yval + 25, yval)
        ctr+=1
    # Show the plot
    plt.show()
    
    start = time.time()
    # # define the undersampling method
    # undersample = NearMiss(version=1, n_neighbors=3)
    # # transform the dataset
    # X2, y2 = undersample.fit_resample(X, y)
    
    # # define the undersampling method
    # undersample = RandomUnderSampler(random_state=42)
    # # transform the dataset
    # X2, y2 = undersample.fit_resample(X, y)
    
    # define the undersampling method
    undersample = EditedNearestNeighbours()
    # transform the dataset
    X2, y2 = undersample.fit_resample(X, y)
    print(f"Undersampling time: {time.time()-start}")
    
    
    # Create bar plot showing the frequency of the different labels
    xlabs, counts = np.unique(y2, return_counts=True)
    bars = plt.bar(xlabs, counts, align='center')
    # Add x labels every 5 bars
    plt.gca().set_xticks(np.arange(0, 53, 5))
    # Add a label to the bar every 5th bar
    ctr = 0
    for bar in bars:
        if ctr % 5 == 0:
            yval = bar.get_height()
            plt.text(bar.get_x(), yval + 0.5, yval)
        ctr+=1
    # Show the plot
    plt.show()
    
    # Split the data 70/20/10
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.33, random_state=42)
    
    
    # Train and predict logisitic regression
    start = time.time()
    clf = LogisticRegression(random_state=0, max_iter=10000).fit(X_train, y_train)
    print(f"LR training time: {time.time()-start}")
    start = time.time()
    predictions = clf.predict(X_test)
    print(f"LR inference time: {(time.time()-start)/len(y_test)}")
    print("Logistic Regression")
    print(classification_report(y_test,predictions))
    probas = clf.predict_proba(X_test)
    multiclass_roc(y_test, probas, 13, "LR ROC Curve")
    
    # Train and predict a SVM
    start = time.time()
    clf = SVC(gamma='auto').fit(X_train, y_train)
    print(f"SVM training time: {time.time()-start}")
    start = time.time()
    predictions = clf.predict(X_test)
    print(f"SVM inference time: {(time.time()-start)/len(y_test)}")
    print("SVM")
    print(classification_report(y_test,predictions))
    probas = label_binarize(predictions, classes=range(0, 13))
    multiclass_roc(y_test, probas, 13, "SVM ROC Curve")
    
    # Train and predict a random forest
    start=time.time()
    clf = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train, y_train)
    print(f"RF training time: {time.time()-start}")
    start = time.time()
    predictions = clf.predict(X_test)
    print(f"RF inference time: {(time.time()-start)/len(y_test)}")
    print("Random Forest")
    print(classification_report(y_test,predictions))
    probas = clf.predict_proba(X_test)
    multiclass_roc(y_test, probas, 13, "RF ROC Curve")
    
    # Define convolutional neural network architecture
    model = models.Sequential()
    # Add fully connected layer
    model.add(layers.Dense(1000, activation='relu', input_shape = (72,)))
    # Add dropout layer drop 20% of connections to prevent overfitting
    model.add(layers.Dropout(0.2))
    # Add a fully connected layer
    model.add(layers.Dense(750, activation='relu'))
    # Add dropout layer drop 20% of connections to prevent overfitting
    model.add(layers.Dropout(0.2))
    # Add a fully connected layer
    model.add(layers.Dense(500, activation='relu'))
    
    
    # Add a fully connected layer for classification
    model.add(layers.Dense(13, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    start = time.time()
    history = model.fit(np.array(X_train),
                        np.array(y_train),
                        epochs=50, 
                        validation_data=(np.array(X_val), 
                                          np.array(y_val)),)
    print(f"FFNN training time: {time.time()-start}")
    
    start = time.time()
    predictions = model.predict(X_test) 
    print(f"FFNN inference time: {(time.time()-start)/len(y_test)}")
    print(classification_report(y_test,np.argmax(predictions, axis=1)))
    multiclass_roc(y_test, predictions, 13, "FFNN ROC Curve")
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()
    
    