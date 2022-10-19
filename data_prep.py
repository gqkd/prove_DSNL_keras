from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import random
from itertools import chain
random.seed(42)

class DataPrep():
    def __init__(self) -> None:
        pass

    #takes files from directory e group them in patients
    def data_loader(self,
                    data_dir = "download_dataset/data",
                    electrods = 'fpz_cz'):

        data_directory = data_dir
        if electrods == 'fpz_cz':
            electrod_directory = "/fpz_cz"
        elif electrods == 'pz_oz':
            electrod_directory = "/pz_oz"

        self.mypath1 = data_directory + electrod_directory
        self.file_list = [f for f in listdir(self.mypath1) if isfile(join(self.mypath1, f))]
        
        self.patient_list = []
        for edf_name in self.file_list:
            self.patient_list.append(edf_name[3:5])
        print(self.patient_list) #only the number of the patient
        
        self.unique_patients = np.unique(np.asarray(self.patient_list))

        #to group the nights of the same patient
        self.group_patient_list = []
        for i in range(len(self.unique_patients)):
            temp=[]
            for j in range(len(self.file_list)):
                if self.unique_patients[i] == self.file_list[j][3:5]:
                    temp.append(self.file_list[j])
            self.group_patient_list.append(temp)
        self.file_list.sort()

    def get_seq(self,seq_length=3):
        data_X, data_y = [], []
        for ind in range(len(self.file_list)):
            with np.load(self.mypath1 + '/' + self.file_list[ind]) as npz:
                data_X.append(npz['x'])
                data_y.append(npz['y'])

        # make sequence data, must be odd
        # seq_length = 3 
        #training
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

        # for k in range(len(data_X)):
        #   for i in range(len(data_X[k])):
        #     if i+seq_length < len(data_X[k]):
        #       tmp90sec=[]
        #       for j in range(seq_length):
        #         seq30sec = list(chain.from_iterable(data_X[k][i+j]))
        #         tmp90sec.append(seq30sec)
        #       X_seq.append(np.concatenate(tmp90sec))
        #       y_seq.append(list(data_y[k][i:i+seq_length]))

        X_seq = np.array(X_seq)
        X_seq = X_seq[:,:,0]
        central_index = round(seq_length/2)-1
        print(central_index)
        y_seq_cen = [y_seq[i][central_index] for i in range(len(y_seq))]

        # temp_ = []
        # for i in range(len(y_seq_train_cen)):
        #     temp = np.zeros((5,))
        #     temp[y_seq_train_cen[i]] = 1.
        #     temp_.append(temp)
        # y_train = np.asarray(temp_)
        y_seq = to_categorical(y_seq_cen,num_classes=len(np.unique(y_seq_cen)))

        return X_seq, y_seq        

    def standard_split( self,
                        split = (75,20,5),
                        raw=False,
                        test=False): #percentage train, validation, test)

        num_pat = len(self.unique_patients)
        print(f"patient number in the list {num_pat}\n")
        print(f"unique patients\n{self.unique_patients}")

        num_pat_train = round(num_pat * split[0]/100)
        num_pat_val = round(num_pat * split[1]/100)
        num_pat_test = round(num_pat * split[2]/100)

        print(f"numero pazienti train {num_pat_train}")
        print(f"numero pazienti val {num_pat_val}")
        print(f"numero pazienti test {num_pat_test}\n")
        
        idx_train = []
        for j in range(num_pat_train):
            idx = [i for i in range(len(self.file_list)) if self.file_list[i][3:5]==self.unique_patients[j]]
            for ind in idx:
                idx_train.append(ind)

        idx_val = []
        for j in range(num_pat_train, num_pat_train + num_pat_val):
            idx = [i for i in range(len(self.file_list)) if self.file_list[i][3:5]==self.unique_patients[j]]
            for ind in idx:
                idx_val.append(ind)
        
        idx_test = []
        for j in range(num_pat_train + num_pat_val, num_pat_train + num_pat_val + num_pat_test):
            idx = [i for i in range(len(self.file_list)) if self.file_list[i][3:5]==self.unique_patients[j]]
            for ind in idx:
                idx_test.append(ind)
        
        print(f"indexes patients in the training set {idx_train}")
        print(f"indexes patients in the validation set {idx_val}")
        print(f"indexes patients in the test set {idx_test}")

        data_X_train, data_y_train = [], []
        for ind in idx_train:
            with np.load(self.mypath1 + '/' + self.file_list[ind]) as npz:
                data_X_train.append(npz['x'])
                data_y_train.append(npz['y'])

        data_X_val, data_y_val = [], []
        for ind in idx_val:
            with np.load(self.mypath1 + '/' + self.file_list[ind]) as npz:
                data_X_val.append(npz['x'])
                data_y_val.append(npz['y'])

        data_X_test, data_y_test = [], []
        for ind in idx_test:
            with np.load(self.mypath1 + '/' + self.file_list[ind]) as npz:
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
    
    def data_down(self, X, y_1HE): #y 1 hot encoded
        y = np.argmax(y_1HE, axis=1)
        uni, counts = np.unique(y, return_counts=True)
        less_represented = np.argmin(counts)
        elements_of_less_represented = counts[less_represented]
        num_classes = len(uni)        
        X_ = []
        y_ = []
        for i in range(num_classes):
            X_temp=[]
            y_temp=[]
            if i==less_represented:
                X_.append(X[y==less_represented])
                y_.append(y_1HE[y==less_represented])
            else:
                X_temp = X[y==i]
                X_.append(random.choices(X_temp, k=elements_of_less_represented))
                y_temp = [to_categorical(i,num_classes) for _ in range(elements_of_less_represented)]
                y_.append(y_temp)
        X_ = np.concatenate(X_)
        y_ = np.array(y_)
        y_ = np.concatenate(y_)

        return X_, y_


    def data_aug(self, X, y):
        y_temp = np.argmax(y, axis=1)
        uni, counts = np.unique(y_temp, return_counts=True)
        most_represented = np.argmax(counts)
        elements_of_most_represented = counts[most_represented]
        num_classes = len(uni)
        X_ = []
        y_ = []

        for i in range(num_classes):
            X1 = []
            X2 = []
            
            if i==most_represented:
                X_.append(X[y_temp==most_represented])
                y_.append(y[y_temp==most_represented])
            else:
                X1.append(X[y_temp==i])
                X1 = np.array(X1)
                X1 = X1[0,:,:]
                X2.append(X[y_temp==i]*-1) #flipped vertically
                X2 = np.array(X2)
                X2 = X2[0,:,:]
                
                X3 = np.concatenate((X1,X2))
                X_.append(X1) #in this way takes the observations and after inserting the original does the random sampling

                X_.append(random.choices(X3, k=elements_of_most_represented-len(X1)))

                # X_.append(X3)
                # X_.append(random.choices(X3, k=elements_of_most_represented-len(X3)))

                y3 = [to_categorical(i,num_classes) for _ in range(elements_of_most_represented)]
                y_.append(y3)

        X_ = np.concatenate(X_)
        y_ = np.array(y_)
        y_ = np.concatenate(y_)
        # y_ = to_categorical(y_)
        return X_, y_

    def data_augN1(self, X,y):
        y_temp = np.argmax(y, axis=1)
        uni, counts = np.unique(y_temp, return_counts=True)
        less_represented = np.argmin(counts)
        # elements_of_less_represented = counts[less_represented]
        num_classes = len(uni)
        X_ = []
        y_ = []

        for i in range(num_classes):
            X1 = []
            X2 = []
            X21 = []
            if i != less_represented:
                X_.append(X[y_temp==i])
                y_.append(y[y_temp==i])
            else:
                X1.append(X[y_temp==i])
                X1 = np.array(X1)
                X1 = X1[0,:,:]
                
                #flip vertically
                X2.append(X[y_temp==i]*-1)
                X2 = np.array(X2)
                X2 = X2[0,:,:]
                
                #rotation backward
                X21.append(X[y_temp==i][::-1])
                X21 = np.array(X21)
                X21 = X21[0,:,:]

                X3 = np.concatenate((X1,X2,X21))

                X_.append(X3)

                # X_.append(X3)
                # X_.append(random.choices(X3, k=elements_of_most_represented-len(X3)))

                y3 = [to_categorical(i,num_classes) for _ in range(len(X3))]
                y_.append(y3)
        X_ = np.concatenate(X_)
        y_ = np.array(y_)
        y_ = np.concatenate(y_)
        return X_, y_
    
    def kfold(self, X, y, k, kth, split=(80,10,10)):
        seq_idx = [i for i in range(len(X))]
        random.shuffle(seq_idx)

        n = int(len(seq_idx)*1/k*kth)
        seq_idx = seq_idx[-n:] + seq_idx[:-n]

        num_train = int(len(X)*split[0]/100)
        num_valid = int(len(X)*split[1]/100)+1
        num_test = int(len(X)*split[2]/100)+1 #no need

        X_seq_train, y_seq_train = [], []
        X_seq_valid, y_seq_valid = [], []
        X_seq_test, y_seq_test = [], []

        for i in range(0, num_train):
            idx = seq_idx[i]
            X_seq_train.append(X[idx])
            y_seq_train.append(y[idx])

        for i in range(num_train, num_train+num_valid):
            idx = seq_idx[i]
            X_seq_valid.append(X[idx])
            y_seq_valid.append(y[idx])

        for i in range(num_train+num_valid, len(seq_idx)):
            idx = seq_idx[i]
            X_seq_test.append(X[idx])
            y_seq_test.append(y[idx])        

        X_seq_train = np.array(X_seq_train)
        y_seq_train = np.array(y_seq_train)

        X_seq_valid = np.array(X_seq_valid)
        y_seq_valid = np.array(y_seq_valid)

        X_seq_test = np.array(X_seq_test)
        y_seq_test = np.array(y_seq_test)

        X_train, y_train = [], []

        for i in range(len(X_seq_train)):
            temp = []
            for j in range(len(X_seq_train[i])):
                temp.append(X_seq_train[i][j])
            X_train.append(temp)
            y_train.append(y_seq_train[i])

        X_valid, y_valid = [], []

        for i in range(len(X_seq_valid)):
            temp = []
            for j in range(len(X_seq_valid[i])):
                temp.append(X_seq_valid[i][j])
            X_valid.append(temp)
            y_valid.append(y_seq_valid[i])

        X_test, y_test = [], []

        for i in range(len(X_seq_test)):
            temp = []
            for j in range(len(X_seq_test[i])):
                temp.append(X_seq_test[i][j])
            X_test.append(temp)
            y_test.append(y_seq_test[i])

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_valid = np.array(X_valid)
        y_valid = np.array(y_valid)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        return X_train, y_train, X_valid, y_valid, X_test, y_test
        
    def kfold_idx(self, X, k, kth, split=(80,10,10)):
        seq_idx = [i for i in range(len(X))]
        random.shuffle(seq_idx)

        n = int(len(seq_idx)*1/k*kth)
        seq_idx = seq_idx[-n:] + seq_idx[:-n]

        num_train = int(len(X)*split[0]/100)
        num_valid = int(len(X)*split[1]/100)+1
        num_test = int(len(X)*split[2]/100)+1 #no need

        idx_train = []
        idx_valid = []
        idx_test = []

        for i in range(0, num_train):
            idx_train.append(seq_idx[i])
        
        for i in range(num_train, num_train+num_valid):
            idx_valid.append(seq_idx[i])
        
        for i in range(num_train+num_valid, len(seq_idx)):
            idx_test.append(seq_idx[i])
     
        return idx_train, idx_valid, idx_test


if __name__ == "__main__":
    dp = DataPrep()
    dp.data_loader()
    # X_train, X_valid, y_train, y_valid = dp.standard_split()
    X, y = dp.get_seq()
    #     print(type(y_train))
    # X_train, X_valid, y_train, y_valid = data_loader()
    # X_train, y_train = data_augN1(X_train, y_train)


