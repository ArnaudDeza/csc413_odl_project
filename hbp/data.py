import numpy as np
import os; os.environ['KERAS_BACKEND'] = 'theano' 
from scipy.io import loadmat

def load(dataset):
    assert dataset in ['higgs', 'cd1','cd2','cd3','cd4','cd5','cd6','cd7','syn8','susy']
    data = loadmat('/u/c/dezaarna/Documents/csc413_odl_project/data/' + dataset + '.mat')
    X_train = data['X_train']
    X_test = data['X_test']
    Y_train = data['Y_train']
    Y_test = data['Y_test']

    nb_classes = 2
    print(X_train.shape)
    return (X_train, Y_train, X_test, Y_test, nb_classes)


def load1(dataset):
    assert dataset in ['higgs', 'cd1','cd2','cd3','cd4','cd5','cd6','cd7','syn8','susy']
    data = np.load('/u/c/dezaarna/Documents/csc413_odl_project/data/' + dataset + '.npz')
    X_train = data['x_train']
   
    Y_train = data['y_train']

    nb_classes = 2
    print(X_train.shape)
    return (X_train, Y_train, nb_classes)

def foo():
    for dataset in ['higgs', 'cd1','cd2','syn8','susy']:
        data = np.load('/u/c/dezaarna/Documents/csc413_odl_project/data/' + dataset + '.npz')
        X_train = data['x_train']
    
        Y_train = data['y_train']
        print(dataset)
        print(X_train.shape)
        print(Y_train.shape)
        print("\n \n")
        
if __name__ == '__main__':
    
    foo()