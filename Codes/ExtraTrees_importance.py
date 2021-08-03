# This script runs ExtarTrees model and returns the average feature importance for 10 runs (with SD) for each descriptor.
import sys,os,re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import ensemble
# Parameters to change
# number of repeats
n_rep=10
# file with descriptors in
loc1_train="train.csv"
train=pd.read_csv(loc1_train)
# descriptors to use in models
descs=["NS_HOMO","NS_LUMO","NS_DM","NS_EB","NS_EA_H","NS_EA_nonH","NS_DeltaG","N_TCA","N_BAD","sol_PCA1","sol_PCA2","sol_PCA3","sol_PCA4","sol_PCA5","N_fukui","N_fukui_Li","N_fukui_charge"]
# name and location of output file
output_file="importance.csv"
# End of parameters to change
# method to run ET and return feature importance
def stat_split_metrics(train,descs):
    # place target value in y
    y_train = train['N']
    # place descriptors in X
    X_train = train[descs]
    # scale data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    # run models
    # ExtraTrees
    tree3 = ensemble.ExtraTreesRegressor(n_estimators=1000,n_jobs=-1)
    tree3.fit(X_train, y_train)
    # return feature importances
    return(tree3.feature_importances_)
# get mean metrics from these predictions and SD as error
# needs to be in pandas dataframe
def get_cons(metrics):
    # Get mean metrics
    lst=[]
    for f in descs:
        temp=[]
        metrics2=metrics[f]
        metrics2=np.array(metrics2)
        mean=np.mean(metrics2)
        std=np.std(metrics2)
        temp.append(f)
        temp.append(mean)
        temp.append(std)
        lst.append(temp)
    mean_metrics=pd.DataFrame(data=lst,columns=["Descriptor","Mean","Std"])
    return(mean_metrics)
# run the model to get importance 
master=[]
for g in range(n_rep):
    features=stat_split_metrics(train,descs)
    master.append(features)
df=pd.DataFrame(data=master,columns=descs)
df=get_cons(df)
df.to_csv(output_file)

