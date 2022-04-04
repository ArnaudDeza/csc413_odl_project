# Links:

ODL repo : https://github.com/LIBOL/ODL?fbclid=IwAR2-4_QBVOZgUWR2Tqr5FViZimknr4cU4UxOsNlL8MiXGaJP-tE8TVW0vCg

ODL paper : https://www.ijcai.org/proceedings/2018/0369.pdf

# Setups:
- Clone repo and then make vitual environment in python 3.6 with library from requirements.txt (Note: add the name of virtual environment folder to .gitignore)
- Download data from google drive link in ODL repo, put all npz files in a folder called "data"
- before activating env, modify the training script of keras as per ODL repo


# Results:

Training results can be found in runs/table2/FINAL_RESULTS.txt --> rougly 45 ish models trained so far

https://github.com/ArnaudDeza/csc413_odl_project/blob/main/runs/table2/FINAL_RESULTS.txt


## Table 3 - Our results
Method | Layers | Higgs | Susy | Syn8
| :---: | :---: | :---: | :---: | :---: 
OGD | 2 | -- | --  | -- 
OGD | 3 | -- | --  | -- 
OGD | 4 | -- | --  | -- 
OGD | 8 | -- | --  | -- 
OGD | 16 | -- | --  | -- 
OGD | 20 | -- | --  | -- 
Hedge BP | 20 | -- | --  | -- 
