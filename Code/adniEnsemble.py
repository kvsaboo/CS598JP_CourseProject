# basic modules
import numpy as np
import pandas as pd
import scipy.io
# learning model modules
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# learning support modules
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from tensorflow import keras
import tensorflow as tf
# learning metrics modules
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
# temporary library
import pdb


def SensiSpeci(true_labels, predicted_labels):
    confmat = confusion_matrix(true_labels, predicted_labels)
    sensitivity = 100*confmat[1,1]/(confmat[1,0]+confmat[1,1]) # TP/(FN+TP)
    specificity = 100*confmat[0,0]/(confmat[0,0]+confmat[0,1]) # TN/(TN+FP)
    return sensitivity, specificity



def gridSearchWrapper(classifier_name, paramdict, num_cv, train_X, train_Y, test_X, test_Y):

    # select and train the model
    if classifier_name.lower() == 'logistic_regression':
        mlmodel = LogisticRegressionCV(Cs=paramdict, cv=num_cv, max_iter=800)
        mlmodel.fit(train_X, train_Y)
    
    elif classifier_name.lower() == 'svm':
        grid_classifier = GridSearchCV(estimator=SVC(), 
                                       param_grid=paramdict, cv=num_cv, verbose=1)
        # train model
        grid_classifier.fit(train_X, train_Y)
        mlmodel = grid_classifier.best_estimator_ # select best model
        
    elif classifier_name.lower() == 'random_forest':
        # grid search for finding best hyper parameters        
        grid_classifier = GridSearchCV(estimator=RandomForestClassifier(),
                                       param_grid=paramdict, cv=num_cv, verbose=1)
        # train model
        grid_classifier.fit(train_X, train_Y)
        mlmodel = grid_classifier.best_estimator_ # select best model
        
    else:
        print("Method not found")
        return 0
    
    # test model - test data
    sensitivity_test, specificity_test = SensiSpeci(test_Y, mlmodel.predict(test_X))
    accuracy_test = 100*mlmodel.score(test_X, test_Y)
    f1_test = f1_score(test_Y, mlmodel.predict(test_X))
    
    # test model - training data
    sensitivity_train, specificity_train = SensiSpeci(train_Y, mlmodel.predict(train_X))
    accuracy_train = 100*mlmodel.score(train_X, train_Y)
    f1_train = f1_score(train_Y, mlmodel.predict(train_X))
    
    # prepare dict to return
    test_score_dict = {'sensitivity':sensitivity_test, 'specificity':specificity_test, 'accuracy':accuracy_test, 'f1':f1_test}
    train_score_dict = {'sensitivity':sensitivity_train, 'specificity':specificity_train, 'accuracy':accuracy_train, 'f1':f1_train}
    
    return {'model': mlmodel, 'test_score':test_score_dict, 'train_score':train_score_dict}



def ttSplitWithGridSearch(classifier_name, paramdict, num_cv, datadf, feature_name, label_name, train_fraction, random_seed):
    # prepare training and test data with appropriate features and randomization
    train_X_ns, test_X_ns, train_Y, test_Y = train_test_split(
        datadf[feature_name], datadf[label_name], train_size=train_fraction, test_size=1-train_fraction,
        random_state=random_seed, shuffle=True)
    
    # standardize features based on mean and standard deviation of each feature in training data
    mean_list_train = np.mean(train_X_ns, axis=0)
    std_list_train = np.std(train_X_ns, axis=0)
    train_X = (train_X_ns-mean_list_train)/std_list_train
    test_X = (test_X_ns-mean_list_train)/std_list_train

    # preform grid search to choose best model
    grid_search_out = gridSearchWrapper(classifier_name, paramdict, num_cv, 
                              train_X, train_Y, test_X, test_Y)
    
    # return best model and its performance
    return grid_search_out

# Deep Learning Model
def trainDeepLearningModelCV(model, data_X, data_Y, val_frac, train_epochs):
    num_cv = int(np.round(1/val_frac))
    val_perf = np.zeros((num_cv, train_epochs))
    train_perf = np.zeros((num_cv, train_epochs))
    
    for cvn in range(0, num_cv):
        print("Training CV number: %d" % (cvn))
        # split training and validation data
        train_X_ns, val_X_ns, train_Y, val_Y = train_test_split(data_X, data_Y, 
                                                          train_size=1-val_frac, test_size=val_frac, shuffle=True)
        # standardize validation and training data using training data
        mean_list_train = np.mean(train_X_ns, axis=0)
        std_list_train = np.std(train_X_ns, axis=0)
        train_X = (train_X_ns - mean_list_train)/std_list_train
        val_X = (val_X_ns - mean_list_train)/std_list_train

        # create model instance and compile it
        temp_model = keras.Sequential.from_config(model.get_config())
        temp_model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='binary_crossentropy',
              metrics=['accuracy'])
        # train model
        temp_model.fit(train_X, train_Y, 
                       epochs=train_epochs, 
                       validation_data=(val_X, val_Y),
                       verbose=0)
        epoch_array = temp_model.history.epoch
        val_perf[cvn,:] = temp_model.history.history['val_acc']
        train_perf[cvn,:] = temp_model.history.history['acc']
    
    return train_perf, val_perf, epoch_array


# train each modality - use the method which had best on average performance for each modality as evaluated in previous experiment
def modalityModelTrainWrapper(classifier_name, modality_name, train_df, num_cv):
    # auxiliary information required in DF
    aux_info = ['Gender','Age','Educ','APOE','DX_bin']
    factors = ['Gender','Age','Educ','APOE']
    
    # grid search parameters
    if classifier_name == 'random_forest':
        param_grid = {'n_estimators':np.arange(10,21,dtype=int), 'max_depth':np.arange(3,7,dtype=int)}
    elif classifier_name == 'logistic_regression':
        param_grid = 20
    else:
        print('Classifier not found')
        return 0
    
    # prepare data
    modality_vars = [fieldname for fieldname in train_df.columns if modality_name in fieldname]
    modality_columns = [item for sublist in [aux_info, modality_vars] for item in sublist]
    modality_features = [item for sublist in [factors, modality_vars] for item in sublist]
    
    train_modalitydf = train_df.loc[train_df[modality_name+'_01'] !=-1, modality_columns]
    
    # standardize training data
    train_modality_mean = train_modalitydf[modality_features].mean()
    train_modality_std = train_modalitydf[modality_features].std()
    train_X_zs = (train_modalitydf[modality_features]-train_modality_mean)/train_modality_std
    
    # perform parameter sweep for classifier and train model
    # NOTE: I'm passing training data as testing in the function since this function is used just to train the model
    gridout = gridSearchWrapper(classifier_name, param_grid, num_cv, train_X_zs, 
                                          train_modalitydf['DX_bin'], train_X_zs, train_modalitydf['DX_bin'])

    return {'model':gridout['model'], 'features':modality_features,
            'train_mod_mean':train_modality_mean, 'train_mod_std':train_modality_std}