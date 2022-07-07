import os
from os import listdir
from os.path import isfile, join
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def data_loader(data_dir = "download_dataset/data", stratified=False):
    data_directory = data_dir
    
    electrod_directory1 = "/eeg_fpz_cz"
    mypath1 = data_directory+electrod_directory1
    file_list1 = [f for f in listdir(mypath1) if isfile(join(mypath1, f))]

    # electrod_directory2 = "/eeg_pz_oz"
    # mypath2 = data_directory+electrod_directory2
    # file_list2 = [f for f in listdir(mypath2) if isfile(join(mypath2, f))]

    data_X, data_y = [], []
    #data for fpz_cz
    for i in range(len(file_list1)):
        with np.load(mypath1+'/'+file_list1[i]) as npz:
            data_X.append(npz['x'])
            data_y.append(npz['y'])

    # temp_y = []
    # for i in range(len(data_y)):
    #     for j in range(len(data_y[i])):
    #         temp_y.append(data_y[i][j])
    # data_y2 = temp_y

    # #data for pz_oz
    # for i in range(len(file_list2)):
    #     with np.load(mypath2+'/'+file_list2[i]) as npz:
    #         data_X.append(npz['x'])
    #         data_y.append(npz['y'])

    # make sequence data, must be odd
    seq_length = 3 
    X_seq, y_seq = [], []

    #in this way is like [ep1,ep2,ep3],[ep4,ep5,ep6], ...
    # for i in range(len(data_X)):
    #     for j in range(0, len(data_X[i]), seq_length): # discard last short sequence
    #         if j+seq_length < len(data_X[i]):
    #             X_seq.append(np.concatenate(data_X[i][j:j+seq_length]))
    #             y_seq.append(list(np.array(data_y[i][j:j+seq_length])))

    #in this way is like [ep1,ep2,ep3],[ep2,ep3,ep4], ...
    for i in range(len(data_X)):
        for j in range(len(data_X[i])): # discard last short sequence
            if j+seq_length < len(data_X[i]):
                X_seq.append(np.concatenate(data_X[i][j:j+seq_length]))
                y_seq.append(list(np.array(data_y[i][j:j+seq_length])))

    X_seq = np.array(X_seq)
    X_seq = X_seq[:,:,0] #to get rid of the useless last dimension
    y_seq_central = [y_seq[i][1] for i in range(len(y_seq))]

    temp_ = []
    for i in range(len(y_seq_central)):
        temp = np.zeros((5,))
        temp[y_seq_central[i]] = 1.
        temp_.append(temp)
    y_seq_central_ohe = np.asarray(temp_)

    
    if stratified:
        #stratified
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, valid_index in sss.split(X_seq, y_seq_central_ohe):
            X_train, X_valid = X_seq[train_index], X_seq[valid_index]
            y_train, y_valid = y_seq_central_ohe[train_index], y_seq_central_ohe[valid_index] 
    else:
        #split train valid
        X_train, X_valid = train_test_split(X_seq, test_size=0.2, random_state=42)
        y_train, y_valid = train_test_split(y_seq_central_ohe, test_size=0.2, random_state=42)

        #how X and y are organized
        # X is a matrix in which each row is a sequence of 3 epochs
        # each epochs is 3000 samples
        # ___________________________
        #| epoch1 + epoch2 + epoch3  |
        #| epoch2 + epoch3 + epoch4  |
        #| ...
        #
        # if the number of epochs per patient is not divisible for 3
        # last epochs are deleted
        # y is a matrix with the class of the epochs
        # __________________________________
        #| [y_epoch1 , y_epoch2 , y_epoch3] |
        #| [y_epoch4 , y_epoch5 , y_epoch6] |
        #| ...
        #
        # we will pass the sequence to the net but we need only the class
        # for the central one
        # _____________
        #|  y_epoch2  |
        #|  y_epoch3  |
        #|  y_epoch4  |
        #| ...


    return X_train, X_valid, y_train, y_valid

