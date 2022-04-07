# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:32:12 2022

@author: Hattie
"""

import os
import scipy.io
import time
import pandas as pd



def reshape_emg_data(emg_data, N, steps, W, H, C):
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
    
    :return: A list of 3d matrices of emg data
    :rtype: list
    """
    # Crete a list to store reshaped emg data images
    output = []
    # Iterate over the emg signal in steps of 100
    for i in range(0, len(emg_data), steps):
        # Get 500 rows of the emg data
        temp = emg_data[i:i+N]
        try:
            # Reshape the emg data to 5x100x12
            img = temp.T.reshape(C, H, W)        
            output.append(img)
        except ValueError as e:
            print(e)
    return output



if __name__ == "__main__":
    print("Hello world")
    
    # List all files in current directory
    files = os.listdir('data/')
    
    # Create empty list to store files and names
    modified_files = []
    
    for file in files:
        if ".mat" in file:
            mat = scipy.io.loadmat(f"data/{file}")    
            images = reshape_emg_data(mat['emg'], 500, 100, 5, 100, 12) 
            
            modified_files.append([file, images])
        