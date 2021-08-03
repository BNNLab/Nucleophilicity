# Script to calculate the optimal number of nodes in a single hidden layer for GAS_SET descriptors.
# import modules
import sys,os,re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from scipy.stats import pearsonr
import math
import statistics
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import DataConversionWarning
warnings.simplefilter('error', category=ConvergenceWarning)
warnings.simplefilter('ignore', category=DataConversionWarning)
# number of layers to examine
n_layers1=np.arange(1,10,1)
n_layers2=np.arange(10,110,10)
n_layers3=np.arange(200,1100,100)
n_layers4=np.arange(2000,6000,1000)
n_layers=[]
n_layers.extend(n_layers1)
n_layers.extend(n_layers2)
n_layers.extend(n_layers3)
n_layers.extend(n_layers4)
n_layers=np.array(n_layers)
# number of repeats for SD error bars
n_rep=10
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
# method to run ANN
# fixed training:test split method
def stat_split_metrics(train,test,n_layers):
    RMSE=[]
    R2=[]
    N1=[]
    N05=[]
    # place target value in y
    y_train = train['N']
    y_test = test['N']
    # place descriptors in X
    X_train = train[["N_E0","N_G0","N_vol","N_HOMO", "N_LUMO","N_DM","N_MW","EB","EA_H","EA_nonH","N_TCA","N_BAD","N_fukui","N_fukui_Li","N_fukui_charge","sol_PCA1","sol_PCA2","sol_PCA3","sol_PCA4","sol_PCA5","sol_PCA_MP","sol_PCA_BP","sol_PCA_Density"]]
    X_test = test[["N_E0","N_G0","N_vol","N_HOMO","N_LUMO","N_DM","N_MW","EB","EA_H","EA_nonH","N_TCA","N_BAD","N_fukui","N_fukui_Li","N_fukui_charge","sol_PCA1","sol_PCA2","sol_PCA3","sol_PCA4","sol_PCA5","sol_PCA_MP","sol_PCA_BP","sol_PCA_Density"]]
    # scale data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # run models
    # ANN
    mlp = MLPRegressor(hidden_layer_sizes=n_layers,max_iter=800)
    mlp.fit(X_train, y_train)   
    mlp2preds = mlp.predict(X_test)
    if np.ptp(mlp2preds) == 0:
        return()
    # evaluate model
    R2.append(pearsonr(mlp2preds, y_test))
    RMSE.append(rmse(mlp2preds, y_test))
    N1.append(within_range(y_test,mlp2preds,2))
    N05.append(within_range(y_test,mlp2preds,1))
    # get R2 from Pearson output
    R2_2=[]
    for i in range(len(R2)):
        x=re.findall('\d\.\d+',str(R2[i]))
        j=float(x[0])
        j=j**2
        R2_2.append(j)
    # create dataframe of metrics
    Metrics=[]
    Metrics.append("ANN")
    Metrics.append(n_layers)
    Metrics.append(R2_2[0])
    Metrics.append(RMSE[0])
    Metrics.append(N1[0])
    Metrics.append(N05[0])
    return(Metrics)
# get mean metrics from these predictions with SD
# needs to be in pandas dataframe
def get_cons(metrics,n_layers):
    # Get mean metrics
    mean_metrics2 = metrics[["R2","RMSE","% within 2","% within 1"]]
    mean_metrics = list(mean_metrics2.mean())
    std_metrics = list(mean_metrics2.std())
    mean_metrics = mean_metrics + std_metrics
    mean_metrics.insert(0,n_layers)
    mean_metrics.insert(0,"ANN")
    return(mean_metrics)
# load datasets
# import train and test set datasets
loc1_train="train.csv"
loc1_test="test.csv"
train_7A=pd.read_csv(loc1_train)
test_7A=pd.read_csv(loc1_test)
# run methods to get metrics for every number of layers and save average metrics (with SD)
# master list
master=[]
# for every architecture
for f in n_layers:
    metrics_all=[]
    for g in range(n_rep):
        try:
            metrics=stat_split_metrics(train_7A,test_7A,f)
        except:
            break
        metrics_all.append(metrics)
    # columns names for intermediate pandas dataframe
    # make dataframe
    metrics_all=pd.DataFrame(data=metrics_all,columns=["Method", "n_layers","R2","RMSE","% within 2","% within 1"])
    if len(metrics_all["R2"])<3:
        continue
    # get metrics of mean predictions
    metrics=get_cons(metrics_all,f)
    master.append(metrics)
metrics_df=pd.DataFrame(data=master,columns=["Method", "n_layers","R2","RMSE","% within 2","% within 1","R2_std","RMSE_std","% 2 std","% 1 std"])
metrics_df.to_csv("ANN_nodes.csv",index=False)
