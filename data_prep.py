from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def data_loader(data_dir = "download_dataset/data",
                electrods = 'fpz_cz',
                raw=False,
                split = (75,20,5), #percentage train, validation, test
                test=False):

    data_directory = data_dir
    if electrods == 'fpz_cz':
        electrod_directory = "/fpz_cz"
    elif electrods == 'pz_oz':
        electrod_directory = "/pz_oz"

    mypath1 = data_directory + electrod_directory
    file_list = [f for f in listdir(mypath1) if isfile(join(mypath1, f))]
    
    #splitting
    patient_list = []
    for edf_name in file_list:
        patient_list.append(edf_name[3:5])
    unique_patients = np.unique(np.asarray(patient_list))
    num_pat = len(unique_patients)
    num_pat_train = round(num_pat * split[0]/100)
    num_pat_val = round(num_pat * split[1]/100)
    num_pat_test = round(num_pat * split[2]/100)

    idx_train = []
    for j in range(num_pat_train):
        idx = [i for i in range(len(file_list)) if file_list[i][3:5]==unique_patients[j]]
        for ind in idx:
            idx_train.append(ind)

    idx_val = []
    for j in range(num_pat_train, num_pat_train + num_pat_val):
        idx = [i for i in range(len(file_list)) if file_list[i][3:5]==unique_patients[j]]
        for ind in idx:
            idx_val.append(ind)
    
    idx_test = []
    for j in range(num_pat_train + num_pat_val, num_pat_train + num_pat_val + num_pat_test):
        idx = [i for i in range(len(file_list)) if file_list[i][3:5]==unique_patients[j]]
        for ind in idx:
            idx_test.append(ind)
        
    data_X_train, data_y_train = [], []
    for ind in idx_train:
        with np.load(mypath1 + '/' + file_list[ind]) as npz:
            data_X_train.append(npz['x'])
            data_y_train.append(npz['y'])

    data_X_val, data_y_val = [], []
    for ind in idx_val:
        with np.load(mypath1 + '/' + file_list[ind]) as npz:
            data_X_val.append(npz['x'])
            data_y_val.append(npz['y'])

    data_X_test, data_y_test = [], []
    for ind in idx_test:
        with np.load(mypath1 + '/' + file_list[ind]) as npz:
            data_X_test.append(npz['x'])
            data_y_test.append(npz['y'])

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
    
    #training
    X_seq_train, y_seq_train = [], []

    #in this way is like [ep1,ep2,ep3],[ep4,ep5,ep6], ...
    # for i in range(len(data_X)):
    #     for j in range(0, len(data_X[i]), seq_length): # discard last short sequence
    #         if j+seq_length < len(data_X[i]):
    #             X_seq.append(np.concatenate(data_X[i][j:j+seq_length]))
    #             y_seq.append(list(np.array(data_y[i][j:j+seq_length])))

    #in this way is like [ep1,ep2,ep3],[ep2,ep3,ep4], ...
    for i in range(len(data_X_train)):
        for j in range(len(data_X_train[i])): # discard last short sequence
            if j+seq_length < len(data_X_train[i]):
                X_seq_train.append(np.concatenate(data_X_train[i][j:j+seq_length]))
                y_seq_train.append(list(np.array(data_y_train[i][j:j+seq_length])))

    X_seq_train = np.array(X_seq_train)
    X_train = X_seq_train[:,:,0] #to get rid of the useless last dimension
    y_seq_train_cen = [y_seq_train[i][1] for i in range(len(y_seq_train))]

    # temp_ = []
    # for i in range(len(y_seq_train_cen)):
    #     temp = np.zeros((5,))
    #     temp[y_seq_train_cen[i]] = 1.
    #     temp_.append(temp)
    # y_train = np.asarray(temp_)
    y_train = to_categorical(y_seq_train_cen)


    #validation
    X_seq_val, y_seq_val = [], []
    for i in range(len(data_X_val)):
        for j in range(len(data_X_val[i])): # discard last short sequence
            if j+seq_length < len(data_X_val[i]):
                X_seq_val.append(np.concatenate(data_X_val[i][j:j+seq_length]))
                y_seq_val.append(list(np.array(data_y_val[i][j:j+seq_length])))

    X_seq_val = np.array(X_seq_val)
    X_val = X_seq_val[:,:,0] #to get rid of the useless last dimension
    y_seq_val_cen = [y_seq_val[i][1] for i in range(len(y_seq_val))]

    # temp_ = []
    # for i in range(len(y_seq_val_cen)):
    #     temp = np.zeros((5,))
    #     temp[y_seq_val_cen[i]] = 1.
    #     temp_.append(temp)
    # y_val = np.asarray(temp_)
    
    y_val = to_categorical(y_seq_val_cen) 
    
    #test
    X_seq_test, y_seq_test = [], []
    for i in range(len(data_X_test)):
        for j in range(len(data_X_test[i])): # discard last short sequence
            if j+seq_length < len(data_X_test[i]):
                X_seq_test.append(np.concatenate(data_X_test[i][j:j+seq_length]))
                y_seq_test.append(list(np.array(data_y_test[i][j:j+seq_length])))

    X_seq_test = np.array(X_seq_test)
    X_test = X_seq_test[:,:,0] #to get rid of the useless last dimension
    y_seq_test_cen = [y_seq_test[i][1] for i in range(len(y_seq_test))]

    # temp_ = []
    # for i in range(len(y_seq_test_cen)):
    #     temp = np.zeros((5,))
    #     temp[y_seq_test_cen[i]] = 1.
    #     temp_.append(temp)
    # y_test = np.asarray(temp_)

    y_test = to_categorical(y_seq_test_cen)

    # if test:
    #     #don't need to shuffle for the test
    #     X_test = X_seq
    #     y_test = y_seq_central_ohe
    # else:
    #     #stratified + shuffle, difference between folds not guaranted
    #     sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    #     for train_index, valid_index in sss.split(X_seq, y_seq_central_ohe):
    #         X_train, X_valid = X_seq[train_index], X_seq[valid_index]
    #         y_train, y_valid = y_seq_central_ohe[train_index], y_seq_central_ohe[valid_index] 

        # #split train valid
        # X_train, X_valid = train_test_split(X_seq, test_size=test_size, random_state=42)
        # y_train, y_valid = train_test_split(y_seq_central_ohe, test_size=test_size, random_state=42)

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

    if raw:
        data = (data_X_train, data_y_train, data_X_val, data_y_val, data_X_test, data_y_test)
        return data
        
    if test:
        return X_test, y_test
    else:
        return X_train, X_val, y_train, y_val

# if __name__ == "__main__":
#     X_train, X_valid, y_train, y_valid = data_loader()