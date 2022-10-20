# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:32:12 2022

@author: Hattie
"""

import os
import random
import scipy.io
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras import layers, models



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


def reshape_emg_data(emg_data, stimulus, N, steps, W, H, C):
    """Function to transform a table of emg signals into a list of 3d matrices
    
    :param emg_data: An array of emg values where each row is reading and each
                     column is a channel
    :type emg_data: numpy.ndarray
    
    :param N: The number of data point to include
    :type N: int
    
    :param steps: The step size between each window
    :type steps: int
    
    :param W: The width/columns of the output matrix
    :type W: int
    
    :param H: The height/rows of the output matrix
    :type H: int
    
    :param C: The number of channels in the emg data
    :type C: int
    
    :return: A list of 3d matrices of emg data and a list of labels
    :rtype: list, list
    """
    # Crete a list to store reshaped emg data images
    output = []
    labels = []
    # Iterate over the emg signal in steps of 100
    for i in range(0, len(emg_data), steps):
        # Get 500 rows of the emg data and labels
        temp = emg_data[i:i+N]
        temp_stimulus = stimulus[i:i+N]
        # If there is a crossover between labels skip feature extraction
        if len(np.unique(temp_stimulus)) > 1:
            continue
        try:
            # Reshape the emg data to WxHxC
            img = temp.reshape(H, W, C)        
            output.append(img)
            # Get the single label value that applies to the window and append
            labels.append(int(np.unique(temp_stimulus)))
        except ValueError:
            continue
        except TypeError:
            continue
    return output, labels


def reshape_emg_data_avg(emg_data, stimulus, N, steps, W, H, C):
    """Function to transform a table of emg signals into a list of 3d matrices
       using the average image value for each label
       
    :param emg_data: An array of emg values where each row is reading and each
                     column is a channel
    :type emg_data: numpy.ndarray
    
    :param N: The number of data point to include
    :type N: int
    
    :param steps: The step size between each window
    :type steps: int
    
    :param W: The width/columns of the output matrix
    :type W: int
    
    :param H: The height/rows of the output matrix
    :type H: int
    
    :param C: The number of channels in the emg data
    :type C: int
    
    :return: A list of 3d matrices of emg data and a list of labels
    :rtype: list, list
    """
    # Get the indexes for each time the label value changes
    indexes = [index for index in range(0, len(stimulus)) if stimulus[index] != stimulus[index-1]]
    # Add the length of the list as the final index 
    indexes.append(len(stimulus))
    
    output, labels = [], []
    
    # Iterate over the list of indexes, excluding the last value
    for i in range(0, len(indexes)-1):
        # Get thew section of emg data with the current label
        temp_emg = emg_data[indexes[i]:indexes[i+1]]
        # Create a list to store all images of the same label
        temp_output = []
        
        # Create counter to count number of images processed
        ctr = 1
        # Iterate over the emg signal in steps of 100
        for j in range(0, len(temp_emg), steps):
            if ctr == 10:
                # Add the average of all images for the given label to the otput
                output.append(sum(temp_output)/len(temp_output))
                # Add current label to the output
                labels.append(stimulus[indexes[i]][0])
                # Reset the temp image list
                temp_output = []
                # Reset the counter 
                ctr = 1
            # Get 500 rows of the emg data and labels
            temp = temp_emg[j:j+N]
            try:
                # Reshape the emg data to WxHxC
                img = temp.reshape(H, W, C)
                # Add the image to the list
                temp_output.append(img)
                # Incremement the counter
                ctr += 1
            except ValueError:
                continue
        try:
            # Add the average of all images for the given label to the otput
            output.append(sum(temp_output)/len(temp_output))
            # Add current label to the output
            labels.append(stimulus[indexes[i]][0])
        except ZeroDivisionError:
            continue
            
    return output, labels


def random_undersample(temp_X, temp_y, amount, val):
    """Functions to remove a number of random values from a dataset
    :param temp_X: A list of data values that correspond to labels
    :type temp_Y: list
    
    :param temp_y: A list of data labels that correspond
    :type temp_y: list
    
    :param amount: The amount of data point to be removed
    :type amount: int
    
    :param val: The label value to be undersampled
    :type val: int
    
    :return: Two resampled lists of data and labels
    :rtype: list, list
    """
    # Get indexes of all labels with desired value
    indexes = list(np.where(np.array(temp_y) == val)[0])
    
    # Get a list of indexes to remove from the dataset
    remove = random.sample(indexes, k=amount)	
    
    # Remove the specified indexes from both lists
    for i in sorted(remove, reverse=True):
        del(temp_X[i])
        del(temp_y[i])
    
    return temp_X, temp_y



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
    
    # Create empty list to store files and names
    modified_files = []
    
    # Define the width, height and channels to be used throughout the script
    W, H, C = 5, 100, 12
    
    start = time.time()
    # Iterate through each file in the list of files
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
            
            # images, labels = reshape_emg_data_avg(mat['emg'], new_stimulus, 500, 100, W, H, C) 
            images, labels = reshape_emg_data(mat['emg'], new_stimulus, 500, 100, W, H, C) 
            
            modified_files.append([file, images, labels])
    
    # Create lists to store all the images and all the labels
    X = []
    y = []
    
    # Iterate through the modified files and extend the image and label lists
    for file in modified_files:
        print(f"Concatentating file {file[0]}")
        X.extend(file[1])
        y.extend(file[2])
    
    print(f"Image processing time: {time.time()-start}")
    
    start = time.time()
    X_norm = []
    img_min = np.min(X)
    img_max = np.max(X)
    for img in X:
        X_norm.append((img-img_min)/(img_max-img_min))
    print(f"Image normalisation time: {time.time()-start}")
    
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
    
    
    amount = int(input("How many?!: "))
    start = time.time()
    # Call function to under sample the data
    X2, y2 = random_undersample(X_norm.copy(), y.copy(), amount, 0)
    print(f"Image under sampling time: {time.time()-start}")
    
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

    print(f"Pre undersampled: {len(X)}")
    print(f"Post undersampled: {len(X2)}")
    

    
    # Split the data 70/20/10
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.33, random_state=42)
    
    # Define convolutional neural network architecture
    model = models.Sequential()
    # Add convolution layer with 64 filters of size 3x3
    model.add(layers.Conv2D(200, (3, 3), activation='relu', input_shape=(H, W, C)))
    # Add dropout layer drop 20% of connections to prevent overfitting
    model.add(layers.Dropout(0.2))
    # Add convolution layer with 128 filters of size 3x3
    model.add(layers.Conv2D(100, (3, 3), activation='relu'))
    # Add a pooling layer
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    # Add convolution layer with 256 filters of size 3x3
    model.add(layers.Conv2D(60, (3, 3), activation='relu'))
    # Add dropout layer drop 20% of connections to prevent overfitting
    model.add(layers.Dropout(0.2))
    # Add convolution layer with 256 filters of size 3x3
    model.add(layers.Conv2D(40, (3, 3), activation='relu'))
    # Add dropout layer drop 20% of connections to prevent overfitting
    model.add(layers.Dropout(0.2))
    # Turn the network from WxHx12 into a single row of data
    model.add(layers.Flatten())
    # Add a fully connected layer of 64 neurons
    model.add(layers.Dense(100, activation='relu'))
    # Add a fully connected layer for classification
    model.add(layers.Dense(13, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    
    start = time.time()
    history = model.fit(np.array(X_train),
                        np.array(y_train),
                        epochs=60, 
                        validation_data=(np.array(X_val), 
                                          np.array(y_val)),)
    
    print(f"CNN training time: {time.time()-start}")
    start = time.time()
    predictions = model.predict(np.array(X_test))
    print(f"CNN inference time: {(time.time()-start)/len(y_test)}")
    print(classification_report(y_test,np.argmax(predictions, axis=1)))
    multiclass_roc(y_test, predictions, 13, "CNN ROC Curve")
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()
