# Script to calculate the optimal numebr of trees in RF for GAS_SET
import sys,os,re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import ensemble
from scipy.stats import pearsonr
import math
import statistics
# number of trees to consider
n_trees1=np.arange(1,10,1)
n_trees2=np.arange(10,110,10)
n_trees3=np.arange(200,1100,100)
n_trees4=np.arange(2000,6000,1000)
n_trees=[]
n_trees.extend(n_trees1)
n_trees.extend(n_trees2)
n_trees.extend(n_trees3)
n_trees.extend(n_trees4)
n_trees=np.array(n_trees)
# number of repeats
n_rep=100
# define metrics
# define RMSE
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
# define % within certain range
def within_range(list1, list2, range2):
    x=0
    for i in range(len(list2)):
        if (list1[i]-range2)<= list2[i] <= (list1[i]+range2): 
            x+=1
    return((float(x)/(len(list2)))*100)
# define getting R2 method
def get_R2(R2):
    R2_2=[]
    for i in range(len(R2)):
        x=re.findall('\d\.\d+',str(R2[i]))
        j=float(x[0])
        j=j**2
        R2_2.append(j)
    return(R2_2)
# method to run ET
# fixed training:test split
def stat_split_metrics(train,test,n_trees):
    RMSE=[]
    R2=[]
    N1=[]
    N05=[]
    # place target value in y
    y_train = train['N']
    y_test = test['N']
    # place descriptors in X
    X_train = train[["N_E0","N_G0","N_vol","N_HOMO","N_LUMO","N_DM","N_MW","EB","EA_H","EA_nonH","N_TCA","N_BAD","N_fukui","N_fukui_Li","N_fukui_charge","sol_PCA1","sol_PCA2","sol_PCA3","sol_PCA4","sol_PCA5","sol_PCA_MP","sol_PCA_BP","sol_PCA_Density"]]
    X_test = test[["N_E0","N_G0","N_vol","N_HOMO","N_LUMO","N_DM","N_MW","EB","EA_H","EA_nonH","N_TCA","N_BAD","N_fukui","N_fukui_Li","N_fukui_charge","sol_PCA1","sol_PCA2","sol_PCA3","sol_PCA4","sol_PCA5","sol_PCA_MP","sol_PCA_BP","sol_PCA_Density"]]
    # scale data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # run models
    # ExtraTrees
    tree3 = ensemble.RandomForestRegressor(n_estimators=n_trees,n_jobs=-1)
    tree3.fit(X_train, y_train)
    tree3preds = tree3.predict(X_test)
    # evaluate model
    R2.append(pearsonr(tree3preds, y_test))
    RMSE.append(rmse(tree3preds, y_test))
    N1.append(within_range(y_test,tree3preds,2))
    N05.append(within_range(y_test,tree3preds,1))
    # get R2 from Pearson output
    R2_2=[]
    for i in range(len(R2)):
        x=re.findall('\d\.\d+',str(R2[i]))
        j=float(x[0])
        j=j**2
        R2_2.append(j)
    # create dataframe of metrics
    Metrics=[]
    Metrics.append("ET")
    Metrics.append(n_trees)
    Metrics.append(R2_2[0])
    Metrics.append(RMSE[0])
    Metrics.append(N1[0])
    Metrics.append(N05[0])
    return(Metrics)
# get mean metrics from these predictions
# needs to be in pandas dataframe
def get_cons(metrics,n_trees):
    # Get mean metrics
    mean_metrics2 = metrics[["R2","RMSE","% within 2","% within 1"]]
    mean_metrics = list(mean_metrics2.mean())
    std_metrics = list(mean_metrics2.std())
    mean_metrics = mean_metrics + std_metrics
    mean_metrics.insert(0,n_trees)
    mean_metrics.insert(0,"RF")
    return(mean_metrics)
# load datasets
# import train and test set datasets
loc1_train="D:\\Nucleophiles\\Descriptor_sets\\Stat_split\\train.csv"
loc1_test="D:\\Nucleophiles\\Descriptor_sets\\Stat_split\\test.csv"
train=pd.read_csv(loc1_train)
test=pd.read_csv(loc1_test)
# run methods
master=[]
for f in n_trees:
    metrics_all=[]
    for g in range(n_rep):
        metrics=stat_split_metrics(train,test,f)
        metrics_all.append(metrics)
    # columns names for intermediate pandas dataframe
    # make dataframe
    metrics_all=pd.DataFrame(data=metrics_all,columns=["Method","n_trees","R2","RMSE","% within 2","% within 1"])
    # get metrics of mean predictions
    metrics=get_cons(metrics_all,f)
    master.append(metrics)
metrics_7A_df=pd.DataFrame(data=master,columns=["Method","n_trees","R2","RMSE","% within 2","% within 1","R2_std","RMSE_std","% 2 std","% 1 std"])
# save metrics
metrics_7A_df.to_csv("RF_trees.csv",index=False)
