# basic modules
import numpy as np
import pandas as pd
import scipy.io
import pickle
import adniEnsemble as adens # module with customized code

# load ADNI dataset and do pre-processing
adnifulldf = pd.read_csv("../Dataset/ADNI/adnitable.csv")
adnifulldf = adnifulldf.fillna(-1)
adnifulldf.DX = adnifulldf.DX.apply(lambda x: 'CN' if x==1 else ('MCI' if x==2 else ('AD' if x==3 else -1)))

## initilizations
# different classifiers and parameters
clf_list = ['logistic_regression','svm','random_forest']
svc_param_grid = {'C':np.logspace(-2,2,5), 'kernel':['linear','poly'], 'degree':[2], 'gamma':[0.1, 1, 10]}
rf_param_grid = {'n_estimators':np.arange(10,21,dtype=int), 'max_depth':np.arange(3,7,dtype=int)}
clf_to_param_dict = {'logistic_regression':20, 'svm':svc_param_grid, 'random_forest':rf_param_grid}
# general params for all classifiers
train_frac = 0.8
gs_num_cv = 4
outer_num_cv = 10
# data related initizations
modality_list = ['amyloid','csf','fdg','pet']
factors = ['Gender','Educ','Age','APOE']
clinical_vars = ['Gender','Educ','Age','APOE','MMSE','DX']
dx_stages = ['CN', 'AD']
# saving results related initiazations
save_dir_path = '../Dataset/ProcessedFiles/'

# run through all data modalities
for modality in modality_list:
    print("Data modality is: " + modality)
    # find columns in entire dataframe related to modality of interest
    modality_vars = [fieldname for fieldname in adnifulldf.columns if modality in fieldname] 
    # create separate dataframe for modality of interest; keep only CN and AD and remove non-APOE subjects
    modalitydf = adnifulldf.loc[(adnifulldf["DX"].isin(dx_stages)) & (adnifulldf["APOE"] != -1) 
                                & (adnifulldf[modality+'_01'] != -1), clinical_vars+modality_vars].copy()
    modalitydf['DX_bin'] = np.where(modalitydf["DX"]=="CN", 0, 1) # CN:0 , AD: 1; required for classification

    # whether or not to include factors like Gender, age and education in classifier
    for include_factors in [0,1]:
        if include_factors == 0: # do not include 'factors' in the features
            features = modality_vars
        else: # include "factors" on the features
            features = [item for sublist in [factors, modality_vars] for item in sublist]
        
        print(features)
        
        # run through all classification models
        for clf_name in clf_list:
            print("Classifier is: " + clf_name)
            grid_out_list = []
            # perform outer_num_cv-fold cross validation
            for cvn in range(0, outer_num_cv):
                # perform train-test split on data and grid search for hyperparameters on training data
                grid_out = adens.ttSplitWithGridSearch(clf_name, clf_to_param_dict[clf_name], gs_num_cv, 
                                              modalitydf, features, 'DX_bin', train_frac, cvn) 
                grid_out_list.append(grid_out)

            # file name to store results in
            if include_factors == 0: # do not include 'factors' in the features
                save_fname = save_dir_path + clf_name +'_'+ modality+'_cv.pckl'
            else: # include "factors" on the features
                save_fname = save_dir_path + clf_name +'_'+ modality+'_factors_cv.pckl'
            
            pickle.dump(grid_out_list, open(save_fname, 'wb'))