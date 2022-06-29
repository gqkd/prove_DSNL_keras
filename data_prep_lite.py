#%%
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import tensorflow as tf

data_directory = "download_dataset/data"
electrod_directory1 = "/eeg_fpz_cz"
electrod_directory2 = "/eeg_pz_oz"
mypath1 = data_directory+electrod_directory1
mypath2 = data_directory+electrod_directory2
file_list1 = [f for f in listdir(mypath1) if isfile(join(mypath1, f))]
file_list2 = [f for f in listdir(mypath2) if isfile(join(mypath2, f))]

data_X, data_y = [], []
#data for fpz_cz
for i in range(len(file_list1)):
    with np.load(mypath1+'/'+file_list1[i]) as npz:
        data_X.append(npz['x'])
        data_y.append(npz['y'])

# #data for pz_oz
# for i in range(len(file_list2)):
#     with np.load(mypath2+'/'+file_list2[i]) as npz:
#         data_X.append(npz['x'])
#         data_y.append(npz['y'])

# one-hot encoding sleep stages
temp_y = []
for i in range(len(data_y)):
    temp_ = []
    for j in range(len(data_y[i])):
        temp = np.zeros((5,))
        temp[data_y[i][j]] = 1.
        temp_.append(temp)
    temp_y.append(np.array(temp_))
data_y = temp_y
#%%
# make sequence data
seq_length = 3 

X_seq, y_seq = [], []

for i in range(len(data_X)):
    for j in range(0, len(data_X[i]), seq_length): # discard last short sequence
        if j+seq_length < len(data_X[i]):
            X_seq.append(np.array(data_X[i][j:j+seq_length]))
            y_seq.append(np.array(data_y[i][j:j+seq_length]))
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# %%
kf = KFold(n_splits=5, random_state=42)


# %%
