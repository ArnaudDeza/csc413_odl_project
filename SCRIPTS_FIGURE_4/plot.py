import matplotlib.pyplot as plt
import numpy as np



def plot_table_2(runs_folder_path):
    to_write = []
    dataset = "higgs" # 'susy','syn8'
    for layer in ["3","4","5","6"]:
        for mode in ["10-15%","60-80%"]:
            try:
                error = np.load("{}/table2/log_mlp{}_{}_{}_error.npy".format(runs_folder_path,layer,dataset,mode))
                
                INFO = "data : {}    num layers :   {}      mode: {}    final error: {}\n".format(dataset,layer,mode,error[-1])
                to_write.append(INFO)
            except:
                pass
    
    
    with open("{}/table2/FINAL_RESULTS.txt".format(runs_folder_path), "w") as text_file:
        for _ in to_write:
            text_file.write(_)
            
    
    
plot_table_2("/u/c/dezaarna/Documents/csc413_odl_project/runs")