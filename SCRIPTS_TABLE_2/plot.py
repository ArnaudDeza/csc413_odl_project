import matplotlib.pyplot as plt
import numpy as np



def plot_table_2(runs_folder_path):
    to_write = []
   
    for dataset in['higgs','susy','syn8']:
        for layer in ["3","4","5","6"]:
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
    
    
    with open("{}/table2/FINAL_RESULTS.txt".format(runs_folder_path), "w") as text_file:
        for _ in to_write:
            text_file.write(_)
            
    
    
plot_table_2("/u/c/dezaarna/Documents/csc413_odl_project/runs")