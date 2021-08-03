# This script performs machine learning for GAS_SET and outputs the predictions and metrics for the fixed training:test split and 10-fold CV Using GP from GPy. Note that the fixed split was run once as GP gives the same model each time it is trained (unlike RF for example).
# import modules required
import sys,os,re
import pandas as pd
import numpy as np
from sklearn import preprocessing
import GPy
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import math
import statistics
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
# define method to find upper and lower predictions within certain range
def within_range_errors(list1, list2, list3, list4, range2):
    x=0
    for i in range(len(list2)):
        if (list1[i]-range2)<= list2[i] <= (list1[i]+range2): 
            x+=1
        elif (list1[i]-range2)<= list3[i] <= (list1[i]+range2): 
            x+=1
        elif (list1[i]-range2)<= list4[i] <= (list1[i]+range2): 
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
# fixed training:test split Method
def stat_split_metrics(train,test,descs):
    RMSE=[]
    R2=[]
    N1=[]
    N05=[]
    N1_e=[]
    N05_e=[]
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
    y_train=[[i] for i in y_train]
    y_train=np.array(y_train)
    # run models
    # GPR
    kernel = GPy.kern.RBF(input_dim=len(descs), variance=1., lengthscale=1.)
    GPR=GPy.models.GPRegression(X_train,y_train,kernel)
    GPR.optimize()
    gpr2preds = GPR.predict(X_test)[0]
    # 1 SD confidence interval (68 % of the prediction range)
    errors=GPR.predict_quantiles(X_test,quantiles=(16,84))
    gpr2preds=[i[0] for i in gpr2preds]
    errors[0]=[i[0] for i in errors[0]]
    errors[1]=[i[0] for i in errors[1]]
    # evaluate model
    R2.append(pearsonr(gpr2preds, y_test))
    RMSE.append(rmse(gpr2preds, y_test))
    N1.append(within_range(y_test,gpr2preds,2))
    N05.append(within_range(y_test,gpr2preds,1))
    N1_e.append(within_range_errors(y_test, gpr2preds,errors[0],errors[1],2))
    N05_e.append(within_range_errors(y_test, gpr2preds,errors[0],errors[1],1))
    #get R2 from Pearson output
    R2_2=[]
    for i in range(len(R2)):
        x=re.findall('\d\.\d+',str(R2[i]))
        j=float(x[0])
        j=j**2
        R2_2.append(j)
    #create dataframe of metrics
    Models=["GPR"]
    Metrics=list(zip(Models,R2_2,RMSE,N1,N05,N1_e,N05_e))
    Metrics_df=pd.DataFrame(data=Metrics, columns=['Model','R2','RMSE','% within 2', '% within 1','Max % within 2','Max % within 1'])
    indiv_preds=list(zip(test['mol_name'],gpr2preds, errors[0],errors[1],y_test))
    indiv_preds_df=pd.DataFrame(data=indiv_preds, columns=["mol_name","GPR","Lower","Upper","Exp"])
    return(Metrics_df,indiv_preds_df)
# define 10-fold CV method
def CV_metrics(Data,folds,descs):
    descs2=[]
    for f in descs:
        descs2.append(f)
    descs2.insert(0,"N")
    # empty df to add to
    preds_df=pd.DataFrame(data=[],columns=["mol_name", "Smiles","N","GPR","Lower","Upper"])
    #initiate lists to add metrics to
    RMSE=[]
    R2=[]
    N1=[]
    N05=[]
    RMSE_std=[]
    R2_std=[]
    N1_std=[]
    N05_std=[]
    GPR_RMSE=[]
    GPR_R2=[]
    GPR_N1=[]
    GPR_N05=[]
    # import Data
    X = Data
    X = X.sample(frac=1).reset_index(drop=True)
    # define k-fold cross validation
    col_names=X.dtypes.index
    X = np.array(X)
    kf = KFold(n_splits=folds)
    for train1, test1 in kf.split(X):
        train=X[train1]
        test=X[test1]
        train=pd.DataFrame(data=train, columns=col_names)
        test=pd.DataFrame(data=test, columns=col_names)
        X_train = train[descs2]
        y_train = train['N']
        X_test = test[descs2]
        y_test = test['N']
        X_train = X_train.drop("N",1)
        X_test = X_test.drop("N",1)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        y_train=[[i] for i in y_train]
        y_train=np.array(y_train)
        # run models
        kernel = GPy.kern.RBF(input_dim=len(descs2)-1, variance=1., lengthscale=1.)
        GPR=GPy.models.GPRegression(X_train,y_train,kernel)
        GPR.optimize()
        gpr2preds = GPR.predict(X_test)[0]
        # 1 SD confidence interval
        errors=GPR.predict_quantiles(X_test,quantiles=(16,84))
        gpr2preds=[i[0] for i in gpr2preds]
        errors[0]=[i[0] for i in errors[0]]
        errors[1]=[i[0] for i in errors[1]]
        # evaluate model
        GPR_R2.append(pearsonr(gpr2preds, y_test))
        GPR_RMSE.append(rmse(gpr2preds, y_test))
        GPR_N1.append(within_range(y_test,gpr2preds,2))
        GPR_N05.append(within_range(y_test,gpr2preds,1))
        # add preds to df
        indiv_preds=list(zip(test['mol_name'],test['Smiles'],test['N'], gpr2preds,errors[0],errors[1]))
        indiv_preds_df=pd.DataFrame(data=indiv_preds, columns=["mol_name","Smiles","N","GPR","Lower","Upper"])
        preds_df=pd.concat([preds_df,indiv_preds_df])
    # get R2 from Pearson output
    GPR_R2=get_R2(GPR_R2)
    # get mean metrics and put together in lists
    R2.append(statistics.mean(GPR_R2))
    RMSE.append(statistics.mean(GPR_RMSE))
    N1.append(statistics.mean(GPR_N1))
    N05.append(statistics.mean(GPR_N05))
    #
    # get std metrics and put together in lists
    R2_std.append(np.std(GPR_R2))
    RMSE_std.append(np.std(GPR_RMSE))
    N1_std.append(np.std(GPR_N1))
    N05_std.append(np.std(GPR_N05))
    #
    # create dataframe of metrics
    Models=["GPR"]
    Metrics=list(zip(Models,R2,R2_std,RMSE,RMSE_std,N1,N1_std,N05,N05_std))
    Metrics_df=pd.DataFrame(data=Metrics, columns=["Model","R2","R2_std","RMSE", "RMSE_std","% Within 2",
                                                   "% Within 2 Std","% Within 1","% Within 1 Std"])
    preds_df=preds_df.sort_values(by=["mol_name"])
    return(Metrics_df,preds_df)
# get average metrics
def average_preds_CV(data):
    metrics=[]
    mets=["GPR","Lower","Upper"]
    for met in mets:
        temp_mets=[]
        for dat in data:
            temp_mets.append(dat[met].tolist())
        temp_mets=np.array(temp_mets)
        temp_mets=np.transpose(temp_mets)
        final_mets_mean=[]
        final_mets_std=[]
        for i in temp_mets:
            final_mets_mean.append(np.mean(i))
        for i in temp_mets:
            final_mets_std.append(np.std(i))
        metrics.append(final_mets_mean)
        metrics.append(final_mets_std)
    metrics=np.array(np.transpose(metrics))
    names=data[0]["mol_name"].tolist()
    N=data[0]["N"].tolist()
    smi=data[0]["Smiles"].tolist()
    metrics=metrics.tolist()
    for j in range(len(names)):
        metrics[j].insert(0,N[j])
        metrics[j].insert(0,smi[j])
        metrics[j].insert(0,names[j])
    columns=["mol_name","Smiles","N","GPR","GPR_std","Lower","Lower_std","Upper","Upper_std"]
    metrics_df=pd.DataFrame(data=metrics,columns=columns)
    return(metrics_df)
# load datasets
# import train and test set datasets
train_loc="train.csv"
test_loc="test.csv"
train=pd.read_csv(train_loc)
test=pd.read_csv(test_loc)
# full dataset
full_loc="full.csv"
full=pd.read_csv(full_loc)
# save file
output="stat_split\\GP_GAS_SET_full.csv"
output_metrics="stat_split\\GP_GAS_SET_full_metrics.csv"
output_CV="CV\\GP_GAS_SET_full.csv"
output_metrics_CV="CV\\GP_GAS_SET_full_metrics.csv"
# descriptors to use in models
descs=["N_MW","N_E0","N_G0","N_vol","N_HOMO","N_LUMO", "N_DM","EB","EA_H","EA_nonH","N_TCA","N_BAD","sol_PCA1",
       "sol_PCA2","sol_PCA3","sol_PCA4","sol_PCA5", "sol_PCA_MP","sol_PCA_BP","sol_PCA_Density","N_fukui","N_fukui_Li","N_fukui_charge"]
# Get metrics and predictions and save
metrics,preds=stat_split_metrics(train,test,descs)
preds.to_csv(output)
metrics.to_csv(output_metrics)
# 10-fold CV
# number of repetitions
rep=10
preds_CV=[]
for f in range(rep):
    metrics_CV,preds=CV_metrics(full,10,descs)
    preds_CV.append(preds)
# get average predictions and save
preds_CV=average_preds_CV(preds_CV)
preds_CV.to_csv(output_CV)
# note that the metrics are from the average of a single run of 10-CV NOT from the average predictions
metrics_CV.to_csv(output_metrics_CV)
