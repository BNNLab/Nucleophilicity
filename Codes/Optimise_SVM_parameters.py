# Script to find the best SVM parameters, shown here for GAS_SET.
# import modules
from sklearn.model_selection import GridSearchCV
import sys,os,re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import svm
from scipy.stats import pearsonr
import math
from sklearn.metrics import r2_score
# define metrics
# define RMSE
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
# define method to find predictions within certain range
def within_range(list1, list2, range2):
    x=0
    for i in range(len(list2)):
        if (list1[i]-range2)<= list2[i] <= (list1[i]+range2): 
            x+=1
    return((float(x)/(len(list2)))*100)
# define fixed training:test split
def stat_split_metrics(train,test,default,C,E,G,descs):
    COD=[]
    RMSE=[]
    R2=[]
    N1=[]
    N05=[]
    # place target value in y
    y_train = train['N']
    y_test = test['N']
    # place descriptors in X
    X_train = train[descs]
    X_test = test[descs]
    # scale data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # run models
    # SVM
    if default=="True":
        svm2 = svm.SVR()
    else:
        svm2 = svm.SVR(C = C, epsilon = E, gamma = G, kernel = 'rbf')
    svm2.fit(X_train, y_train)
    svm2preds = svm2.predict(X_test)
    # evaluate model
    COD.append(r2_score(svm2preds, y_test))
    R2.append(pearsonr(svm2preds, y_test))
    RMSE.append(rmse(svm2preds, y_test))
    N1.append(within_range(y_test,svm2preds,2))
    N05.append(within_range(y_test,svm2preds,1))
    # get R2 from Pearson output
    R2_2=[]
    for i in range(len(R2)):
        x=re.findall('\d\.\d+',str(R2[i]))
        j=float(x[0])
        j=j**2
        R2_2.append(j)
    # create dataframe of metrics
    Metrics=[]
    Metrics.append("SVM")
    Metrics.append(COD[0])
    Metrics.append(R2_2[0])
    Metrics.append(RMSE[0])
    Metrics.append(N1[0])
    Metrics.append(N05[0])
    Metrics_df=pd.DataFrame(data=[Metrics],columns=["Method", "COD","R2","RMSE","% within 2","% within 1"])
    indiv_preds=list(zip(test['mol_name'],svm2preds))
    indiv_preds_df=pd.DataFrame(data=indiv_preds, columns=["Name","SVM"])
    return(Metrics_df,indiv_preds_df)
# grid search method
def grid_search(train,param_grid,descs):
    # place target value in y
    y_train = train['N']
    # place descriptors in X
    X_train = train[descs]
    # scale data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    # grid search
    svm2 = svm.SVR()
    gs=GridSearchCV(estimator=svm2,param_grid=param_grid,cv=10)
    gs=gs.fit(X_train,y_train)
	# return best parameters
    return(gs.best_params_)
# import train and test set
loc1_train="train.csv"
loc1_test="test.csv"
train=pd.read_csv(loc1_train)
test=pd.read_csv(loc1_test)
# descriptors to use in the model
descs=["N_E0","N_G0","N_vol","N_HOMO","N_LUMO","N_DM","N_MW","EB","EA_H","EA_nonH","N_TCA","N_BAD","N_fukui","N_fukui_Li","N_fukui_charge",
       "sol_PCA1","sol_PCA2","sol_PCA3","sol_PCA4","sol_PCA5","sol_PCA_MP","sol_PCA_BP","sol_PCA_Density"]
# values to examine in grid search
epsilon_values=[0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
gamma_values=[0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
C_values=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# set up grid
param_grid1={'kernel':['rbf'],'epsilon':epsilon_values,'gamma':gamma_values,'C':C_values}
# get best parameters for only training set
best_params1=grid_search(train,param_grid1,descs)
# print results
print(best_params1)
# now test again default on test set
# default
metrics_d,preds_d=stat_split_metrics(train,test,"True",0,0,0,descs)
print(metrics_d)
# grid search parameters
metrics_p,preds_p=stat_split_metrics(train,test,"False",best_params1["C"],best_params1["epsilon"],best_params1["gamma"],descs)
print(metrics_p)
