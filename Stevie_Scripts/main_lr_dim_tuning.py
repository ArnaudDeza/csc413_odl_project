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
              'data': 'higgs',
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
        
        
        
        
        
        
        
        
        
    base_folder = "/u/c/dezaarna/Documents/csc413_odl_project/sensitivity"
    
    
    
    
    
    
    
    
    
    
    
    
    
    try:
      os.mkdir("{}/{}".format(base_folder,"hidden_dim"))
    except:
      pass
    try:
      os.mkdir("{}/{}".format(base_folder,"learning_rate"))
    except:
      pass
    (X_train, Y_train, _) = load("higgs")
      
    
    hidden_dim = [10,50,100,150,200]
    lrs = [1e-6,1e-5,0.0001,0.001,0.01,0.1,1]
    
   
    
    for tuning_lr in [False, True]:
        if tuning_lr == False:
          curr_folder = "{}/{}".format(base_folder,"hidden_dim")
          for dim in hidden_dim:
                config['hidden_num'] = dim
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
                np.save("{}/dim_{}_error.npy".format(curr_folder,dim),cumAverageError)
                np.save("{}/dim_{}_acc.npy".format(curr_folder,dim),cumAverageAcc)
                np.save("{}/dim_{}_loss.npy".format(curr_folder,dim),cumAverageLoss)
                
                np.save("{}/dim_{}_error_per_itter.npy".format(curr_folder,dim),my_callback_error)
                np.save("{}/dim_{}_acc_per_itter.npy".format(curr_folder,dim),my_callback.acc)
                np.save("{}/dim_{}_loss_per_itter.npy".format(curr_folder,dim),my_callback.l)
        
        if tuning_lr:
          curr_folder = "{}/{}".format(base_folder,"learning_rate")
          for lr_ in lrs:
                
                model, in_name, out_name = build_model(config)
                if len(out_name) == 1:
                    out_name_loss = ['loss']
                else:
                    out_name_loss = [s + '_loss' for s in out_name]

                model.summary()
                
                plot(model, to_file = 'model.png')
                
                optim = eval(config['optim'])(lr = lr_)
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
                np.save("{}/lr_{}_error.npy".format(curr_folder,lr_),cumAverageError)
                np.save("{}/lr_{}_acc.npy".format(curr_folder,lr_),cumAverageAcc)
                np.save("{}/lr_{}_loss.npy".format(curr_folder,lr_),cumAverageLoss)
                
                np.save("{}/lr_{}_error_per_itter.npy".format(curr_folder,lr_),my_callback_error)
                np.save("{}/lr_{}_acc_per_itter.npy".format(curr_folder,lr_),my_callback.acc)
                np.save("{}/lr_{}_loss_per_itter.npy".format(curr_folder,lr_),my_callback.l)
        
   
    
if __name__ == '__main__':
    my_callback = main(sys.argv[1:], 0)
