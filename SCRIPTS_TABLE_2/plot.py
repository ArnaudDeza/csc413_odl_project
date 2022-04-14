import matplotlib.pyplot as plt
import numpy as np



def plot_table_2(runs_folder_path):
    to_write = []
   
    for dataset in['higgs','susy','syn8','cd1','cd2']:
        for layer in ["2","3","4","8","16","20"]:
            for mode in ["full"]:
                try:
                    error = np.load("{}/table2/log_mlp{}_{}_{}_error.npy".format(runs_folder_path,layer,dataset,mode))
                    Zero_One_Acc_per_itter = np.load("{}/table2/log_mlp{}_{}_{}_error_per_itter.npy".format(runs_folder_path,layer,dataset,mode))
                  
                    
                    predictions_0_005 = Zero_One_Acc_per_itter[0:int(0.005*len(Zero_One_Acc_per_itter))]
                    predictions_10_15 = Zero_One_Acc_per_itter[int(0.1*len(Zero_One_Acc_per_itter)):int(0.15*len(Zero_One_Acc_per_itter))]
                    predictions_60_80 = Zero_One_Acc_per_itter[int(0.6*len(Zero_One_Acc_per_itter)):int(0.8*len(Zero_One_Acc_per_itter))]
                    
                    indexes1, indexes2,indexes3 = np.arange(len(predictions_0_005))+1,np.arange(len(predictions_10_15))+1,np.arange(len(predictions_60_80))+1
                    
                    cumAvgAcc_0_005 = np.cumsum(predictions_0_005)/indexes1
                    cumAvgAcc_10_15 = np.cumsum(predictions_10_15)/indexes2
                    cumAvgAcc_60_80 = np.cumsum(predictions_60_80)/indexes3
                    
                    INFO = "data : {}    num layers :   {}       final error: {}\n".format(dataset,layer,error[-1])
                    to_write.append(INFO)
                    INFO = "data : {}    num layers :   {}       0-0.05: {}\n".format(dataset,layer,cumAvgAcc_0_005[-1])
                    to_write.append(INFO)
                    INFO = "data : {}    num layers :   {}       10-15: {}\n".format(dataset,layer,cumAvgAcc_10_15[-1])
                    to_write.append(INFO)
                    INFO = "data : {}    num layers :   {}       60-80: {}\n".format(dataset,layer,cumAvgAcc_60_80[-1])
                    to_write.append(INFO)
                    
                except:
                    pass
    for dataset in['higgs','susy','syn8','cd1','cd2']:
        for modee in ["momentum","nesterov"]:
            for mode in ["full"]:
                try:
                    error = np.load("{}/table2/{}_{}_{}_error.npy".format(runs_folder_path,modee,dataset,mode))
                    Zero_One_Acc_per_itter = np.load("{}/table2/{}_{}_{}_error_per_itter.npy".format(runs_folder_path,modee,dataset,mode))
                  
                    
                    predictions_0_005 = Zero_One_Acc_per_itter[0:int(0.005*len(Zero_One_Acc_per_itter))]
                    predictions_10_15 = Zero_One_Acc_per_itter[int(0.1*len(Zero_One_Acc_per_itter)):int(0.15*len(Zero_One_Acc_per_itter))]
                    predictions_60_80 = Zero_One_Acc_per_itter[int(0.6*len(Zero_One_Acc_per_itter)):int(0.8*len(Zero_One_Acc_per_itter))]
                    
                    indexes1, indexes2,indexes3 = np.arange(len(predictions_0_005))+1,np.arange(len(predictions_10_15))+1,np.arange(len(predictions_60_80))+1
                    
                    cumAvgAcc_0_005 = np.cumsum(predictions_0_005)/indexes1
                    cumAvgAcc_10_15 = np.cumsum(predictions_10_15)/indexes2
                    cumAvgAcc_60_80 = np.cumsum(predictions_60_80)/indexes3
                    
                    INFO = "data : {}    mode :   {}       final error: {}\n".format(dataset,modee,error[-1])
                    to_write.append(INFO)
                    INFO = "data : {}    mode :   {}       0-0.05: {}\n".format(dataset,modee,cumAvgAcc_0_005[-1])
                    to_write.append(INFO)
                    INFO = "data : {}    mode :   {}       10-15: {}\n".format(dataset,modee,cumAvgAcc_10_15[-1])
                    to_write.append(INFO)
                    INFO = "data : {}    mode :   {}       60-80: {}\n".format(dataset,modee,cumAvgAcc_60_80[-1])
                    to_write.append(INFO)
                    
                except:
                    pass
    
    with open("{}/table2/FINAL_RESULTS.txt".format(runs_folder_path), "w") as text_file:
        for _ in to_write:
            text_file.write(_)
            
def plot_dim(runs_folder_path):
    to_write = []
    
   
    for dataset in['higgs','cd1']:
        for layer in ["4","8","16","20"]:
            folder = "{}/log_mlp{}_{}".format(runs_folder_path,layer,dataset)
            for lr in [10,50,100,150,200]:
                try:
                    error = np.load("{}/dim_{}_error.npy".format(folder,lr))
                    
                    Zero_One_Acc_per_itter = np.load("{}/dim_{}_error_per_itter.npy".format(folder,lr))
                        
                    
                        
                    predictions_0_005 = Zero_One_Acc_per_itter[0:int(0.005*len(Zero_One_Acc_per_itter))]
                    predictions_10_15 = Zero_One_Acc_per_itter[int(0.1*len(Zero_One_Acc_per_itter)):int(0.15*len(Zero_One_Acc_per_itter))]
                    predictions_60_80 = Zero_One_Acc_per_itter[int(0.6*len(Zero_One_Acc_per_itter)):int(0.8*len(Zero_One_Acc_per_itter))]
                    
                    indexes1, indexes2,indexes3 = np.arange(len(predictions_0_005))+1,np.arange(len(predictions_10_15))+1,np.arange(len(predictions_60_80))+1
                    
                    cumAvgAcc_0_005 = np.cumsum(predictions_0_005)/indexes1
                    cumAvgAcc_10_15 = np.cumsum(predictions_10_15)/indexes2
                    cumAvgAcc_60_80 = np.cumsum(predictions_60_80)/indexes3
                    
                    INFO = "data : {}    num layers :   {}    dim :      {}   final error: {}\n".format(dataset,layer,lr,error[-1])
                    to_write.append(INFO)
                    INFO = "data : {}    num layers :   {}    dim :      {}   0-0.05: {}\n".format(dataset,layer,lr,cumAvgAcc_0_005[-1])
                    to_write.append(INFO)
                    INFO = "data : {}    num layers :   {}    dim :      {}   10-15: {}\n".format(dataset,layer,lr,cumAvgAcc_10_15[-1])
                    to_write.append(INFO)
                    INFO = "data : {}    num layers :   {}    dim :      {}   60-80: {}\n".format(dataset,layer,lr,cumAvgAcc_60_80[-1])
                    to_write.append(INFO)
                    
                except:
                        pass
    folder = "/u/c/dezaarna/Documents/csc413_odl_project/Sensitivity/sensitivity_hidden_dim/log_hbp19_cd1"
    for lr in [10,50,100,150,200]:
        error = np.load("{}/dim_{}_error.npy".format(folder,lr))
                        
        Zero_One_Acc_per_itter = np.load("{}/dim_{}_error_per_itter.npy".format(folder,lr))
            
        
            
        predictions_0_005 = Zero_One_Acc_per_itter[0:int(0.005*len(Zero_One_Acc_per_itter))]
        predictions_10_15 = Zero_One_Acc_per_itter[int(0.1*len(Zero_One_Acc_per_itter)):int(0.15*len(Zero_One_Acc_per_itter))]
        predictions_60_80 = Zero_One_Acc_per_itter[int(0.6*len(Zero_One_Acc_per_itter)):int(0.8*len(Zero_One_Acc_per_itter))]
        
        indexes1, indexes2,indexes3 = np.arange(len(predictions_0_005))+1,np.arange(len(predictions_10_15))+1,np.arange(len(predictions_60_80))+1
        
        cumAvgAcc_0_005 = np.cumsum(predictions_0_005)/indexes1
        cumAvgAcc_10_15 = np.cumsum(predictions_10_15)/indexes2
        cumAvgAcc_60_80 = np.cumsum(predictions_60_80)/indexes3
        
        INFO = "data : {}    num layers :   {}    dim :      {}   final error: {}\n".format("cd1","hbp",lr,error[-1])
        to_write.append(INFO)
        INFO = "data : {}    num layers :   {}    dim :      {}   0-0.05: {}\n".format("cd1","hbp",lr,cumAvgAcc_0_005[-1])
        to_write.append(INFO)
        INFO = "data : {}    num layers :   {}    dim :      {}   10-15: {}\n".format("cd1","hbp",lr,cumAvgAcc_10_15[-1])
        to_write.append(INFO)
        INFO = "data : {}    num layers :   {}    dim :      {}   60-80: {}\n".format("cd1","hbp",lr,cumAvgAcc_60_80[-1])
        to_write.append(INFO)
    
    with open("{}/FINAL_RESULTS_dim.txt".format(runs_folder_path), "w") as text_file:
        for _ in to_write:
            text_file.write(_)   























def plot_lr(runs_folder_path):
    to_write = []
    
   
    for dataset in['higgs','cd1']:
        for layer in ["4","8","16","20"]:
            folder = "{}/log_mlp{}_{}".format(runs_folder_path,layer,dataset)
            for lr in [1e-6,1e-5,0.0001,0.001,0.01,0.1,1]:
                try:
                    error = np.load("{}/lr_{}_error.npy".format(folder,lr))
                    
                    Zero_One_Acc_per_itter = np.load("{}/lr_{}_error_per_itter.npy".format(folder,lr))
                        
                    
                        
                    predictions_0_005 = Zero_One_Acc_per_itter[0:int(0.005*len(Zero_One_Acc_per_itter))]
                    predictions_10_15 = Zero_One_Acc_per_itter[int(0.1*len(Zero_One_Acc_per_itter)):int(0.15*len(Zero_One_Acc_per_itter))]
                    predictions_60_80 = Zero_One_Acc_per_itter[int(0.6*len(Zero_One_Acc_per_itter)):int(0.8*len(Zero_One_Acc_per_itter))]
                    
                    indexes1, indexes2,indexes3 = np.arange(len(predictions_0_005))+1,np.arange(len(predictions_10_15))+1,np.arange(len(predictions_60_80))+1
                    
                    cumAvgAcc_0_005 = np.cumsum(predictions_0_005)/indexes1
                    cumAvgAcc_10_15 = np.cumsum(predictions_10_15)/indexes2
                    cumAvgAcc_60_80 = np.cumsum(predictions_60_80)/indexes3
                    
                    INFO = "data : {}    num layers :   {}    lr :      {}   final error: {}\n".format(dataset,layer,lr,error[-1])
                    to_write.append(INFO)
                    INFO = "data : {}    num layers :   {}    lr :      {}   0-0.05: {}\n".format(dataset,layer,lr,cumAvgAcc_0_005[-1])
                    to_write.append(INFO)
                    INFO = "data : {}    num layers :   {}    lr :      {}   10-15: {}\n".format(dataset,layer,lr,cumAvgAcc_10_15[-1])
                    to_write.append(INFO)
                    INFO = "data : {}    num layers :   {}    lr :      {}   60-80: {}\n".format(dataset,layer,lr,cumAvgAcc_60_80[-1])
                    to_write.append(INFO)
                    
                except:
                        pass
    folder = "/u/c/dezaarna/Documents/csc413_odl_project/Sensitivity/sensitivity_analysis/log_hbp19_cd1"
    for lr in [1e-6,1e-5,0.0001]:
        
            error = np.load("{}/lr_{}_error.npy".format(folder,lr))
            
            Zero_One_Acc_per_itter = np.load("{}/lr_{}_error_per_itter.npy".format(folder,lr))
                
            
                
            predictions_0_005 = Zero_One_Acc_per_itter[0:int(0.005*len(Zero_One_Acc_per_itter))]
            predictions_10_15 = Zero_One_Acc_per_itter[int(0.1*len(Zero_One_Acc_per_itter)):int(0.15*len(Zero_One_Acc_per_itter))]
            predictions_60_80 = Zero_One_Acc_per_itter[int(0.6*len(Zero_One_Acc_per_itter)):int(0.8*len(Zero_One_Acc_per_itter))]
            
            indexes1, indexes2,indexes3 = np.arange(len(predictions_0_005))+1,np.arange(len(predictions_10_15))+1,np.arange(len(predictions_60_80))+1
            
            cumAvgAcc_0_005 = np.cumsum(predictions_0_005)/indexes1
            cumAvgAcc_10_15 = np.cumsum(predictions_10_15)/indexes2
            cumAvgAcc_60_80 = np.cumsum(predictions_60_80)/indexes3
            
            INFO = "data : {}    num layers :   {}    lr :      {}   final error: {}\n".format("cd1","hbp",lr,error[-1])
            to_write.append(INFO)
            INFO = "data : {}    num layers :   {}    lr :      {}   0-0.05: {}\n".format("cd1","hbp",lr,cumAvgAcc_0_005[-1])
            to_write.append(INFO)
            INFO = "data : {}    num layers :   {}    lr :      {}   10-15: {}\n".format("cd1","hbp",lr,cumAvgAcc_10_15[-1])
            to_write.append(INFO)
            INFO = "data : {}    num layers :   {}    lr :      {}   60-80: {}\n".format("cd1","hbp",lr,cumAvgAcc_60_80[-1])
            to_write.append(INFO)               
   
                    
    with open("{}/FINAL_RESULTS_LEARNING_RATE.txt".format(runs_folder_path), "w") as text_file:
        for _ in to_write:
            text_file.write(_)
            
    

    
plot_table_2("/u/c/dezaarna/Documents/csc413_odl_project/runs")
#plot_lr("/u/c/dezaarna/Documents/csc413_odl_project/Sensitivity/sensitivity_analysis")    
#plot_dim("/u/c/dezaarna/Documents/csc413_odl_project/Sensitivity/sensitivity_hidden_dim")    
