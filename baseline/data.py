import numpy as np
import os 
from scipy.io import loadmat
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.utils import np_utils
from keras import backend as K

def load(dataset):
    print(dataset)
    assert dataset in ['higgs', 'cd1','cd2','syn8','susy']
    data = np.load('/u/c/dezaarna/Documents/csc413_odl_project/data/' + dataset + '.npz')
    X_train = data['x_train']
   
    Y_train = data['y_train']

    nb_classes = 2
    print(X_train.shape)
    return (X_train, Y_train, nb_classes)