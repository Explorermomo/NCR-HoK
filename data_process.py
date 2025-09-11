import time
import os
import random
import warnings
import torch
import numpy as np
import argparse
import scipy.io as sio
import torch.nn as nn

def data_process(train_val_mat, test_mat, batch_size):
    num_train_ins = len(train_val_mat['data_train'])
    num_valid_ins = len(train_val_mat['data_x'])
    num_test_ins = len(test_mat['test'])
    shuffle_list_train = random.sample(range(0, num_train_ins), num_train_ins)

    A_train = []
    y_train = []
    for i in shuffle_list_train:
        tmpA = train_val_mat['data_train'][shuffle_list_train[i], 0]['A'][0, 0].todense()
        tmpy = np.squeeze(train_val_mat['data_train'][shuffle_list_train[i], 0]['y'][0, 0])
        A_train.append(np.expand_dims(tmpA, axis=2))
        y_train.append(tmpy)

    A_val = []
    y_val = []
    for i in range(num_valid_ins):
        tmpA = train_val_mat['data_x'][i, 0]['A'][0, 0].todense()
        tmpy = np.squeeze(train_val_mat['data_x'][i, 0]['y'][0, 0])
        A_val.append(np.expand_dims(tmpA, axis=2))
        y_val.append(tmpy)

    A_test = []
    y_test = []
    for i in range(num_valid_ins):
        tmpA = num_test_ins['data_x'][i, 0]['A'][0, 0].todense()
        tmpy = np.squeeze(num_test_ins['data_x'][i, 0]['y'][0, 0])
        A_test.append(np.expand_dims(tmpA, axis=2))
        y_test.append(tmpy)

    return A_train,y_train,A_val,y_val,A_test,y_test


# load training data
def load_batch(batch_size, begin_index,mat,shuffle_list):
    A = []
    y = []
    for i in range(batch_size):
        tmpA = mat['data_train'][shuffle_list[i+begin_index], 0]['A'][0, 0].todense()
        tmpy = np.squeeze(mat['data_train'][shuffle_list[i + begin_index], 0]['y'][0, 0])
        A.append(np.expand_dims(tmpA, axis=2))
        y.append(tmpy)
    return A, y


# load cross validation data
def load_batch_val(batch_size, begin_index,mat):
    A = []
    y = []
    for i in range(batch_size):
        tmpA = mat['data_x'][i + begin_index, 0]['A'][0, 0].todense()
        tmpy = np.squeeze(mat['data_x'][i + begin_index, 0]['y'][0, 0])
        A.append(np.expand_dims(tmpA, axis=2))
        y.append(tmpy)
    return A, y

def load_mat(idx, mat):
    A = []
    y = []
    tmpA = mat['test'][idx, 0]['A'][0, 0].todense()
    tmpy = np.squeeze(mat['test'][idx, 0]['y'][0, 0])
    A.append(np.expand_dims(tmpA, axis=2))
    y.append(tmpy)
    return A, y