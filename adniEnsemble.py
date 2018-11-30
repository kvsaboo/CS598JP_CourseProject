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
# learning metrics modules
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix



def SensiSpeci(true_labels, predicted_labels):
    confmat = confusion_matrix(true_labels, predicted_labels)
    sensitivity = 100*confmat[1,1]/(confmat[1,0]+confmat[1,1]) # TP/(FN+TP)
    specificity = 100*confmat[0,0]/(confmat[0,0]+confmat[0,1]) # TN/(TN+FP)
    return sensitivity, specificity



def gridSearchWrapper(classifier_name, paramdict, num_cv, train_X, train_Y, test_X, test_Y):

    # select and train the model
    if classifier_name.lower() == 'logistic_regression':
        mlmodel = LogisticRegressionCV(Cs=paramdict, cv=num_cv, max_iter=500)
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
    train_X, test_X, train_Y, test_Y = train_test_split(
        datadf[feature_name], datadf[label_name], train_size=train_fraction, test_size=1-train_fraction,
        random_state=random_seed, shuffle=True)
    
    # preform grid search to choose best model
    grid_search_out = gridSearchWrapper(classifier_name, paramdict, num_cv, 
                              train_X, train_Y, test_X, test_Y)
    
    # return best model and its performance
    return grid_search_out

