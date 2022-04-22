# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:32:12 2022

@author: Hattie
"""

import os
import scipy.io
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models


def reshape_emg_data(emg_data, repetition, N, steps, W, H, C):
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
        temp_repetition = repetition[i:i+N]
        # If there is a crossover between labels skip feature extraction
        if len(np.unique(temp_repetition)) > 1:
            continue
        try:
            # Reshape the emg data to WxHxC
            img = temp.reshape(H, W, C)        
            output.append(img)
            # Get the single label value that applies to the window and append
            labels.append(int(np.unique(temp_repetition)))
        except ValueError as e:
            print(e)
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


if __name__ == "__main__":
    print("Hello world")
    
    # List all files in current directory
    files = os.listdir('data/')
    
    # Create empty list to store files and names
    modified_files = []
    
    for file in files:
        if ".mat" in file:
            mat = scipy.io.loadmat(f"data/{file}")    
            images, labels = reshape_emg_data(mat['emg'], mat['repetition'], 500, 100, 5, 100, 12) 
            
            modified_files.append([file, images, labels])
    
    # Create lists to store all the images and all the labels
    X = []
    y = []
    
    # Iterate through the modified files and extend the image and label lists
    for file in modified_files:
        X.extend(file[1])
        y.extend(file[2])
        
    # Create bar plot showing the frequency of the different labels
    xlabs, counts = np.unique(y, return_counts=True)
    bars = plt.bar(xlabs, counts, align='center')
    plt.gca().set_xticks(xlabs)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + 100, yval)
    plt.show()
    
    X2, y2 = random_undersample(X.copy(), y.copy(), 14000, 0)
    
    # Create bar plot showing the frequency of the different labels
    xlabs, counts = np.unique(y2, return_counts=True)
    bars = plt.bar(xlabs, counts, align='center')
    plt.gca().set_xticks(xlabs)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + 100, yval)
    plt.show()
        
    # Split the data 70/20/10
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.33, random_state=42)
    
    # Define the model input
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(100, 5, 12)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(7, activation='softmax'))
    
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    history = model.fit(np.array(X_train), np.array(y_train), epochs=4, validation_data=(np.array(X_val), np.array(y_val)))
    
    test_loss, test_acc = model.evaluate(np.array(X_test), np.array(y_test), verbose=2)
    print(test_acc)