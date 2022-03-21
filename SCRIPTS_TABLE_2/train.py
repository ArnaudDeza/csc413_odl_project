
import numpy as np
import os, sys, getopt 
import yaml
import os; os.environ['KERAS_BACKEND'] = 'theano' 
import matplotlib.pyplot as plt
import keras
import keras.callbacks
from keras.datasets import mnist
from keras.utils.visualize_util import plot
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Dropout, Highway
from keras.optimizers import SGD, Adam
from keras.callbacks import Callback
import keras.backend as K
import time



def get_data(config):
    if config['data'] in ['syn8','cd1','cd2']:
        config['input_size'] = (50,)
        config['output_size'] = 2
    elif config['data'] == 'higgs':
        config['input_size'] = (28,)
        config['output_size'] = 2
    elif config['data'] == 'susy':
        config['input_size'] = (18,)
        config['output_size'] = 2
    elif config['data'] == 'cd6' or config['data'] == 'cd7':
        config['input_size'] = (50,)
    elif config['data'] == 'cd3' or config['data'] == 'cd4':
        config['input_size'] = (25,)
    config['output_size'] = 2
    return config

def build_model(config):
    config = get_data(config)

    base_name = 'out'
    if config['hedge'] == True:
        outs = ['']*config['n_layers']
        out_name = ['']*config['n_layers']
        N = config['n_layers']
        for i in range(len(outs)):
            outs[i] = base_name + str(i)
            out_name[i] = base_name + str(i)
    else:
        outs = base_name
        out_name = [base_name]
        N = config['n_layers'] - 1
    in_name = 'in0'

    inputs = Input(config['input_size'], name = in_name)
    
    for j in range(N):
        if j == 0:
            layer = Dense(config['hidden_num'])(inputs)
            layer = Activation(config['activation'])(layer)

            if config['hedge'] == True:
                outs[j] = Dense(config['output_size'], activation = 'softmax', name = outs[j])(layer)
            continue
        if config['Highway'] == False:
            layer = Dense(config['hidden_num'])(layer)
            layer = Activation(config['activation'])(layer)
        else:
            layer = Highway(activation = config['activation'])(layer)
            
        if config['hedge'] == True:
            outs[j] = Dense(config['output_size'], activation = 'softmax', name = outs[j])(layer)
    if config['hedge'] == False:
        outs = Dense(config['output_size'], activation = 'softmax', name = outs)(layer)
    model = Model(input = inputs , output = outs)

    return (model, in_name, out_name)

# add self.masks, self.weighted_losses
class MyCallback(Callback):
    def __init__(self,w,  beta = 0.9,  names = [], hedge = False, log_name = None):
        self.weights = w
        self.beta = beta
        self.names = names
        self.l = []
        self.hedge = hedge
        self.acc = []
        self.log_name = log_name
    def on_batch_end(self, batch, logs = {}):
        self.l.append(logs.get('loss'))
        losses = [logs[name] for name in self.names]
        self.acc.append(logs.get('acc'))
        if self.hedge:

            M = sum(losses)
            losses = [loss / M for loss in losses]
            min_loss = np.amin(losses)
            max_loss = np.amax(losses)
            range_of_loss = max_loss - min_loss
            losses = [(loss-min_loss)/range_of_loss for loss in losses]

            alpha = [self.beta ** loss for loss in losses]
            
            try:
                alpha = [a * w for a, w in zip(alpha, self.weights)]
            except ValueError:
                pass

            alpha = [ max(0., a) for a in alpha]
            M = sum(alpha)
            alpha = [a / M for a in alpha]
            
            self.weights = alpha 
    def on_batch_begin(self, epoch, logs={}):
        self.model.holder = (self.weights)
        #print(self.model.holder)

def load(dataset):
    print(dataset)
    assert dataset in ['higgs', 'cd1','cd2','syn8','susy']
    data = np.load('/u/c/dezaarna/Documents/csc413_odl_project/data/' + dataset + '.npz')
    X_train = data['x_train']
   
    Y_train = data['y_train']

    nb_classes = 2
    return (X_train, Y_train, nb_classes)



def build_data_dict(in_name, out_name, in_data, out_data):
    in_dict = dict()
    in_dict[in_name] = in_data
    
    out_dict = dict((k, out_data) for k in out_name)
    return (in_dict, out_dict)

def build_loss_weight(config):
    if config['hedge'] == False:
        w = [1.]
    elif config['loss_weight'] == 'ave':
        w = [1./ config['n_layers']]* config['n_layers']
    return w
def main(arg,mode, idx=0):
    config = {'learning_rate': 1e-3,
              'optim': 'Adam',
              'batch_size': 1,
              'nb_epoch': 50,
              'n_layers': 3,
              'hidden_num': 100,
              'activation': 'relu',
              'loss_weight': 'ave',
              'adaptive_weight': False,
              'data': 'mnist',
              'hedge': False,
              'Highway': False,
              'momentum': 0.,
              'nesterov': False,
              'log': 'mnist_hedge.log'}

    configfile = ''
    helpstring = 'main.py -c <config YAML file>'
    try:
        opts, args = getopt.getopt(arg, "hc:", ["config"])
    except getopt.GetoptError:
        print(helpstring)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print (helpstring)
            yamlstring = yaml.dump(config,default_flow_style=False,explicit_start=True)
            print("YAML configuration file format:")
            print("")
            print("%YAML 1.2")
            print(yamlstring)
            sys.exit()

        elif opt in ('-c', '--config'):
            configfile = arg

        print("Config file is %s" % configfile)

    if os.path.exists(configfile):
        f = open(configfile)
        user_config = yaml.load(f.read())
        config.update(user_config)
    
    print("Printing configuration:")
    for key,value in config.items():
        print("  ",key,": ",value)

    (X_train, Y_train,_) = load(config['data'])
    
    model, in_name, out_name = build_model(config)
    if len(out_name) == 1:
        out_name_loss = ['loss']
    else:
        out_name_loss = [s + '_loss' for s in out_name]

    model.summary()
    
    #plot(model, to_file = 'model.png')
    
    optim = eval(config['optim'])(lr = config['learning_rate'], momentum = config['momentum'], nesterov = config['nesterov'])
    in_dict, out_dict = build_data_dict(in_name, out_name, X_train, Y_train)
    loss_dict = dict((k, 'categorical_crossentropy') for k in out_name) 
  
    loss_weights = build_loss_weight(config)
    my_callback = MyCallback(loss_weights, names = out_name_loss, hedge = config['hedge'], log_name = config['log'])
    #csv  = CSVLogger(config['log'])
    model.compile(optimizer = optim, loss = loss_dict, hedge = config['hedge'],loss_weights = loss_weights, metrics = ['accuracy'])
    model.fit(in_dict, out_dict, nb_epoch = config['nb_epoch'], batch_size = config['batch_size'], callbacks=[my_callback])
    
    
    
    cumAcc, cumLoss = np.cumsum(my_callback.acc),np.cumsum(my_callback.l)
    
    indexOfAcc,indexOfLoss =  np.arange(len(cumAcc))+1,np.arange(len(cumLoss))+1
    cumAverageAcc,cumAverageLoss = cumAcc/indexOfAcc,cumLoss/indexOfLoss
    
    cumAverageError = np.ones_like(cumAverageAcc) -cumAverageAcc
    my_callback_error = np.ones_like(my_callback.acc) -my_callback.acc
    
    plt.close()
    plt.plot(list(range(len(cumAverageAcc))),cumAverageAcc)
    plt.savefig("/u/c/dezaarna/Documents/csc413_odl_project/runs/table2/{}_{}_acc.png".format(config['log'],mode))
    plt.close()
    plt.plot(list(range(len(cumAverageLoss))),cumAverageLoss)
    plt.savefig("/u/c/dezaarna/Documents/csc413_odl_project/runs/table2/{}_{}_loss.png".format(config['log'],mode))
    plt.close()
    plt.plot(list(range(len(cumAverageError))),cumAverageError)
    plt.savefig("/u/c/dezaarna/Documents/csc413_odl_project/runs/table2/{}_{}_error.png".format(config['log'],mode))
    
    '''filename = (config['log'] + '_' + str(idx) + '.acc')
    np.savetxt(filename, cumAverageAcc, delimiter=',')
    
    filename = (config['log'] + '_' + str(idx) + '.loss')
    np.savetxt(filename, cumAverageLoss, delimiter=',')
    
    filename = (config['log'] + '_' + str(idx) + '.error')
    np.savetxt(filename, cumAverageError, delimiter=',')'''
    
    
    np.save("/u/c/dezaarna/Documents/csc413_odl_project/runs/table2/{}_{}_error.npy".format(config['log'],mode),cumAverageError)
    np.save("/u/c/dezaarna/Documents/csc413_odl_project/runs/table2/{}_{}_acc.npy".format(config['log'],mode),cumAverageAcc)
    np.save("/u/c/dezaarna/Documents/csc413_odl_project/runs/table2/{}_{}_loss.npy".format(config['log'],mode),cumAverageLoss)
    
    np.save("/u/c/dezaarna/Documents/csc413_odl_project/runs/table2/{}_{}_error_per_itter.npy".format(config['log'],mode),my_callback_error)
    np.save("/u/c/dezaarna/Documents/csc413_odl_project/runs/table2/{}_{}_acc_per_itter.npy".format(config['log'],mode),my_callback.acc)
    np.save("/u/c/dezaarna/Documents/csc413_odl_project/runs/table2/{}_{}_loss_per_itter.npy".format(config['log'],mode),my_callback.l)
    

if __name__ == '__main__':

    main(sys.argv[1:],"full",idx = 0)
