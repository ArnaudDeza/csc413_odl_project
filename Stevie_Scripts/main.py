import os; os.environ['KERAS_BACKEND'] = 'theano' 
import os, sys, getopt 
import yaml
import numpy as np
import keras
import keras.callbacks
from keras.datasets import mnist
from keras.utils.visualize_util import plot
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam, RMSprop
from model import build_model, MyCallback
from keras.callbacks import CSVLogger
from data import load
import os
os.environ["KERAS_BACKEND"] = "theano"
import keras.backend
keras.backend.set_image_dim_ordering('th')
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
def main(arg, idx=0):
    config = {'learning_rate': 0.01,
              'optim': 'SGD',
              'batch_size': 1,
              'nb_epoch': 1,
              'n_layers': 19,
              'hidden_num': 100,
              'activation': 'relu',
              'loss_weight': 'ave',
              'adaptive_weight': False,
              'data': 'cd1',
              'model': 'MLP',
              'hedge': True,
              'input_size': 50,
              'log': 'log_hbp19'}



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
        
        
    base_folder = "/u/c/dezaarna/Documents/csc413_odl_project/runss"
        
        
        
    for dataset_ in ['higgs','susy','syn8','cd1','cd2']:
        
        
        config['data'] = dataset_
        config['log'] = 'log_hbp19_{}'.format(dataset_)
        
        (X_train, Y_train, _) = load(dataset_)
        
        
        
        X_train = X_train[:300]
        Y_train = Y_train[:300]
        
        
        
        
        curr_folder = "{}/{}".format(base_folder,config['log'])
        try:
                os.mkdir(curr_folder)
        except:
                pass
            
        
        model, in_name, out_name = build_model(config)
        if len(out_name) == 1:
            out_name_loss = ['loss']
        else:
            out_name_loss = [s + '_loss' for s in out_name]

        model.summary()
        
        plot(model, to_file = 'model.png')
        
        optim = eval(config['optim'])(lr = config['learning_rate'])
        in_dict, out_dict = build_data_dict(in_name, out_name, X_train, Y_train)
        #in_val, out_val = build_data_dict(in_name, out_name, X_test, Y_test)
        loss_dict = dict((k, 'categorical_crossentropy') for k in out_name) 
    
        loss_weights = build_loss_weight(config)
        my_callback = MyCallback(loss_weights, names = out_name_loss, hedge = config['hedge'], log_name = config['log'])
        #csv  = CSVLogger(config['log'])
        model.compile(optimizer = optim, loss = loss_dict, hedge = config['hedge'],loss_weights = loss_weights, metrics = ['accuracy'])
        model.fit(in_dict, out_dict, nb_epoch = config['nb_epoch'], batch_size = config['batch_size'], callbacks=[my_callback])
        
            
        cumAcc, cumLoss = np.cumsum(my_callback.acc),np.cumsum(my_callback.l)
        
        indexOfAcc,indexOfLoss =  np.arange(len(cumAcc))+1,np.arange(len(cumLoss))+1
        
        cumAverageAcc = cumAcc/indexOfAcc
        cumAverageLoss = cumLoss/indexOfLoss
        
        cumAverageError = np.ones_like(cumAverageAcc) -cumAverageAcc
        my_callback_error = np.ones_like(my_callback.acc) -my_callback.acc
        np.save("{}/error.npy".format(curr_folder),cumAverageError)
        np.save("{}/acc.npy".format(curr_folder),cumAverageAcc)
        np.save("{}/loss.npy".format(curr_folder),cumAverageLoss)
        
        np.save("{}/error_per_itter.npy".format(curr_folder),my_callback_error)
        np.save("{}/acc_per_itter.npy".format(curr_folder),my_callback.acc)
        np.save("{}/loss_per_itter.npy".format(curr_folder),my_callback.l)
        
        
    
if __name__ == '__main__':
    #for i in range(5):
    my_callback = main(sys.argv[1:], 0)
