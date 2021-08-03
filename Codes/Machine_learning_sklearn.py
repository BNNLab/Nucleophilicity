# This script performs machine learning for GAS_SET and outputs the predictions and metrics for the fixed training:test split and 10-fold CV.
# import modules required
import sys,os,re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.cross_decomposition import PLSRegression
from sklearn import ensemble
from scipy.stats import pearsonr
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import statistics
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import DataConversionWarning
warnings.simplefilter('error', category=ConvergenceWarning)
warnings.simplefilter('ignore', category=DataConversionWarning)
# Definitions of metrics used
# define RMSE
def rmse(predictions, targets):
    predictions=np.array(predictions)
    targets=np.array(targets)
    return np.sqrt(((predictions - targets) ** 2).mean())
# define method to find predictions within certain range
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
# Fixed training:test method "stat split"
def stat_split_metrics(train,test,C,E,G,descs):
    # just get the descriptors required
    train1=train[descs]
    test1=test[descs]
    RMSE=[]
    R2=[]
    N1=[]
    N05=[]
    # place target value in y
    y_train = train1['N'].tolist()
    y_test = test1['N'].tolist()
    # place descriptors in X, i.e. remove "N"
    X_train = train1.drop('N',1)
    X_test = test1.drop('N',1)
    # scale data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # run models
    # MLR
    mlr = LinearRegression()
    mlr.fit(X_train, y_train)
    mlr2preds = mlr.predict(X_test)
    # evaluate model
    R2.append(pearsonr(mlr2preds, y_test))
    RMSE.append(rmse(mlr2preds, y_test))
    N1.append(within_range(y_test,mlr2preds,2))
    N05.append(within_range(y_test,mlr2preds,1))
    # ANN, try 100 times to converge, max interations 800
    # hidden layer size optimised
    mlp = MLPRegressor(hidden_layer_sizes=500,max_iter=800)
    for f in range(100):
        try:
            mlp.fit(X_train, y_train)
            mlp2preds = mlp.predict(X_test)
            if np.ptp(mlp2preds) == 0:
                continue
            break
        except:
            continue
    # evaluate model
    R2.append(pearsonr(mlp2preds, y_test))
    RMSE.append(rmse(mlp2preds, y_test))
    N1.append(within_range(y_test,mlp2preds,2))
    N05.append(within_range(y_test,mlp2preds,1))
    # SVM
    # parameters found via grid search
    svm2 = svm.SVR(C = C, epsilon = E, gamma = G, kernel = 'rbf')
    svm2.fit(X_train, y_train)
    svm2preds = svm2.predict(X_test)
    # evaluate model
    R2.append(pearsonr(svm2preds, y_test))
    RMSE.append(rmse(svm2preds, y_test))
    N1.append(within_range(y_test,svm2preds,2))
    N05.append(within_range(y_test,svm2preds,1))
    # PLS
    pls2 = PLSRegression(n_components=9)
    pls2.fit(X_train, y_train)
    pls2preds = pls2.predict(X_test)
    # evaluate model
    # convert to float (comes in weird type)
    pls2preds2=[]
    for i in pls2preds:
        pls2preds2.append(float(i))
    R2.append(pearsonr(pls2preds2, y_test))
    RMSE.append(rmse(pls2preds2, y_test))
    N1.append(within_range(y_test,pls2preds2,2))
    N05.append(within_range(y_test,pls2preds2,1))
    # RF
    # number of trees (n_estimators) optimisied
    tree2 = ensemble.RandomForestRegressor(n_estimators=1000,n_jobs=-1)
    tree2.fit(X_train, y_train)
    tree2preds = tree2.predict(X_test)
    # evaluate model
    R2.append(pearsonr(tree2preds, y_test))
    RMSE.append(rmse(tree2preds, y_test))
    N1.append(within_range(y_test,tree2preds,2))
    N05.append(within_range(y_test,tree2preds,1))
    # ExtraTrees
    tree3 = ensemble.ExtraTreesRegressor(n_estimators=1000,n_jobs=-1)
    tree3.fit(X_train, y_train)
    tree3preds = tree3.predict(X_test)
    # evaluate model
    R2.append(pearsonr(tree3preds, y_test))
    RMSE.append(rmse(tree3preds, y_test))
    N1.append(within_range(y_test,tree3preds,2))
    N05.append(within_range(y_test,tree3preds,1))
    # Bagging
    tree4 = ensemble.BaggingRegressor(n_estimators=1000,n_jobs=-1)
    tree4.fit(X_train, y_train)
    tree4preds = tree4.predict(X_test)
    # evaluate model
    R2.append(pearsonr(tree4preds, y_test))
    RMSE.append(rmse(tree4preds, y_test))
    N1.append(within_range(y_test,tree4preds,2))
    N05.append(within_range(y_test,tree4preds,1))
    # get R2 from Pearson output
    R2_2=[]
    for i in range(len(R2)):
        x=re.findall('\d\.\d+',str(R2[i]))
        j=float(x[0])
        j=j**2
        R2_2.append(j)
    # create dataframe of metrics
    Models=["MLR","ANN","SVM","PLS","RF","ExtraTrees","Bagging"]
    Metrics=list(zip(Models,R2_2,RMSE,N1,N05))
    Metrics_df=pd.DataFrame(data=Metrics, columns=['Model','R2','RMSE','% within 2','% within 1'])
    indiv_preds=list(zip(test['mol_name'],mlr2preds, mlp2preds,svm2preds,pls2preds2,tree2preds,tree3preds,tree4preds))
    indiv_preds_df=pd.DataFrame(data=indiv_preds, columns=["mol_name","MLR","ANN","SVM","PLS","RF","ExtraTrees","Bagging"])
    return(Metrics_df,indiv_preds_df)
# Method to perform cross validation
def CV_metrics(Data,folds,C,E,G,descs):
    # empty df to add to
    preds_df=pd.DataFrame(data=[],columns=["mol_name", "Smiles","N","MLR","ANN","SVM","PLS","RF","ExtraTrees","Bagging"])
    # initiate lists to add metrics to
    RMSE=[]
    R2=[]
    N1=[]
    N05=[]
    RMSE_std=[]
    R2_std=[]
    N1_std=[]
    N05_std=[]
    MLR_RMSE=[]
    MLR_R2=[]
    MLR_N1=[]
    MLR_N05=[]
    ANN_RMSE=[]
    ANN_R2=[]
    ANN_N1=[]
    ANN_N05=[]
    SVM_RMSE=[]
    SVM_R2=[]
    SVM_N1=[]
    SVM_N05=[]
    PLS_RMSE=[]
    PLS_R2=[]
    PLS_N1=[]
    PLS_N05=[]
    RF_RMSE=[]
    RF_R2=[]
    RF_N1=[]
    RF_N05=[]
    ET_RMSE=[]
    ET_R2=[]
    ET_N1=[]
    ET_N05=[]
    BG_RMSE=[]
    BG_R2=[]
    BG_N1=[]
    BG_N05=[]
    #import Data
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
        X_train = train[descs]
        y_train = train['N']
        X_test = test[descs]
        y_test = test['N']
        X_train = X_train.drop("N",1)
        X_test = X_test.drop("N",1)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        # run models
        # MLR
        mlr = LinearRegression()
        mlr.fit(X_train, y_train)
        mlr2preds = mlr.predict(X_test)
        #evaluate model
        MLR_R2.append(pearsonr(mlr2preds, y_test))
        MLR_RMSE.append(rmse(mlr2preds, y_test))
        MLR_N1.append(within_range(y_test,mlr2preds,2))
        MLR_N05.append(within_range(y_test,mlr2preds,1))
        # ANN
        mlp = MLPRegressor(hidden_layer_sizes=600,max_iter=800)
        for f in range(100):
            try:
                mlp.fit(X_train, y_train)
                mlp2preds = mlp.predict(X_test)
                if np.ptp(mlp2preds) == 0:
                    continue
                break
            except:
                continue
        # evaluate model
        ANN_R2.append(pearsonr(mlp2preds, y_test))
        ANN_RMSE.append(rmse(mlp2preds, y_test))
        ANN_N1.append(within_range(y_test,mlp2preds,2))
        ANN_N05.append(within_range(y_test,mlp2preds,1))
        # SVM
        svm2 = svm.SVR(C = C, epsilon = E, gamma = G, kernel = 'rbf')
        svm2.fit(X_train, y_train)
        svm2preds = svm2.predict(X_test)
        # evaluate model
        SVM_R2.append(pearsonr(svm2preds, y_test))
        SVM_RMSE.append(rmse(svm2preds, y_test))
        SVM_N1.append(within_range(y_test,svm2preds,2))
        SVM_N05.append(within_range(y_test,svm2preds,1))
        # PLS
        pls2 = PLSRegression(n_components=9)
        pls2.fit(X_train, y_train)
        pls2preds = pls2.predict(X_test)
        # evaluate model
        # convert to float (comes in weird type)
        pls2preds2=[]
        for i in pls2preds:
            pls2preds2.append(float(i))
        PLS_R2.append(pearsonr(pls2preds2, y_test))
        PLS_RMSE.append(rmse(pls2preds2, y_test))
        PLS_N1.append(within_range(y_test,pls2preds2,2))
        PLS_N05.append(within_range(y_test,pls2preds2,1))
        # RF
        tree2 = ensemble.RandomForestRegressor(n_estimators=1000,n_jobs=-1)
        tree2.fit(X_train, y_train)
        tree2preds = tree2.predict(X_test)
        # evaluate model
        RF_R2.append(pearsonr(tree2preds, y_test))
        RF_RMSE.append(rmse(tree2preds, y_test))
        RF_N1.append(within_range(y_test,tree2preds,2))
        RF_N05.append(within_range(y_test,tree2preds,1))
        # ExtraTrees
        tree3 = ensemble.ExtraTreesRegressor(n_estimators=1000,n_jobs=-1)
        tree3.fit(X_train, y_train)
        tree3preds = tree3.predict(X_test)
        # evaluate model
        ET_R2.append(pearsonr(tree3preds, y_test))
        ET_RMSE.append(rmse(tree3preds, y_test))
        ET_N1.append(within_range(y_test,tree3preds,2))
        ET_N05.append(within_range(y_test,tree3preds,1))
        # Bagging
        tree4 = ensemble.BaggingRegressor(n_estimators=1000,n_jobs=-1)
        tree4.fit(X_train, y_train)
        tree4preds = tree4.predict(X_test)
        # evaluate model
        BG_R2.append(pearsonr(tree4preds, y_test))
        BG_RMSE.append(rmse(tree4preds, y_test))
        BG_N1.append(within_range(y_test,tree4preds,2))
        BG_N05.append(within_range(y_test,tree4preds,1))
        # add preds to df
        indiv_preds=list(zip(test['mol_name'],test['Smiles'],test['N'],mlr2preds,mlp2preds,svm2preds,pls2preds2,tree2preds,tree3preds,tree4preds))
        indiv_preds_df=pd.DataFrame(data=indiv_preds, columns=["mol_name","Smiles","N","MLR","ANN","SVM","PLS","RF","ExtraTrees","Bagging"])
        preds_df=pd.concat([preds_df,indiv_preds_df])
    # get R2 from Pearson output
    MLR_R2=get_R2(MLR_R2)
    ANN_R2=get_R2(ANN_R2)
    SVM_R2=get_R2(SVM_R2)
    PLS_R2=get_R2(PLS_R2)
    RF_R2=get_R2(RF_R2)
    ET_R2=get_R2(ET_R2)
    BG_R2=get_R2(BG_R2)
    # get mean metrics and put together in lists
    R2.append(statistics.mean(MLR_R2))
    RMSE.append(statistics.mean(MLR_RMSE))
    N1.append(statistics.mean(MLR_N1))
    N05.append(statistics.mean(MLR_N05))
    #
    R2.append(statistics.mean(ANN_R2))
    RMSE.append(statistics.mean(ANN_RMSE))
    N1.append(statistics.mean(ANN_N1))
    N05.append(statistics.mean(ANN_N05))
    #
    R2.append(statistics.mean(SVM_R2))
    RMSE.append(statistics.mean(SVM_RMSE))
    N1.append(statistics.mean(SVM_N1))
    N05.append(statistics.mean(SVM_N05))
    #
    R2.append(statistics.mean(PLS_R2))
    RMSE.append(statistics.mean(PLS_RMSE))
    N1.append(statistics.mean(PLS_N1))
    N05.append(statistics.mean(PLS_N05))
    #
    R2.append(statistics.mean(RF_R2))
    RMSE.append(statistics.mean(RF_RMSE))
    N1.append(statistics.mean(RF_N1))
    N05.append(statistics.mean(RF_N05))
    #
    R2.append(statistics.mean(ET_R2))
    RMSE.append(statistics.mean(ET_RMSE))
    N1.append(statistics.mean(ET_N1))
    N05.append(statistics.mean(ET_N05))
    #
    R2.append(statistics.mean(BG_R2))
    RMSE.append(statistics.mean(BG_RMSE))
    N1.append(statistics.mean(BG_N1))
    N05.append(statistics.mean(BG_N05))
    #get std metrics and put together in lists
    R2_std.append(np.std(MLR_R2))
    RMSE_std.append(np.std(MLR_RMSE))
    N1_std.append(np.std(MLR_N1))
    N05_std.append(np.std(MLR_N05))
    #
    R2_std.append(np.std(ANN_R2))
    RMSE_std.append(np.std(ANN_RMSE))
    N1_std.append(np.std(ANN_N1))
    N05_std.append(np.std(ANN_N05))
    #
    R2_std.append(np.std(SVM_R2))
    RMSE_std.append(np.std(SVM_RMSE))
    N1_std.append(np.std(SVM_N1))
    N05_std.append(np.std(SVM_N05))
    #
    R2_std.append(np.std(PLS_R2))
    RMSE_std.append(np.std(PLS_RMSE))
    N1_std.append(np.std(PLS_N1))
    N05_std.append(np.std(PLS_N05))
    #
    R2_std.append(np.std(RF_R2))
    RMSE_std.append(np.std(RF_RMSE))
    N1_std.append(np.std(RF_N1))
    N05_std.append(np.std(RF_N05))
    #
    R2_std.append(np.std(ET_R2))
    RMSE_std.append(np.std(ET_RMSE))
    N1_std.append(np.std(ET_N1))
    N05_std.append(np.std(ET_N05))
    #
    R2_std.append(np.std(BG_R2))
    RMSE_std.append(np.std(BG_RMSE))
    N1_std.append(np.std(BG_N1))
    N05_std.append(np.std(BG_N05))
    #
    # create dataframe of metrics
    Models=["MLR","ANN","SVM","PLS","RF","ExtraTrees","Bagging"]
    Metrics=list(zip(Models,R2,R2_std,RMSE,RMSE_std,N1,N1_std,N05,N05_std))
    Metrics_df=pd.DataFrame(data=Metrics, columns=["Model","R2","R2_std","RMSE","RMSE_std","% Within 2","% Within 2 Std","% Within 1","% Within 1 Std"])
    preds_df=preds_df.sort_values(by=["mol_name"])
    return(Metrics_df,preds_df)
# get average metrics
def average_metrics(data):
    metrics=[]
    mets=["R2","RMSE","% within 2","% within 1"]
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
    names=data[0]["Model"].tolist()
    metrics=metrics.tolist()
    for j in range(len(names)):
        metrics[j].insert(0,names[j])
    columns=["Model","R2","R2_std","RMSE","RMSE_std","% Within 2", "% Within 2 Std","% Within 1", "% Within 1 Std"]
    metrics_df=pd.DataFrame(data=metrics,columns=columns)
    return(metrics_df)
# get average predictions
def average_preds(data):
    metrics=[]
    mets=["MLR","ANN","SVM","PLS","RF","ExtraTrees","Bagging"]
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
    metrics=metrics.tolist()
    for j in range(len(names)):
        metrics[j].insert(0,names[j])
    columns=["mol_name","MLR","MLR_std","ANN","ANN_std", "SVM","SVM_std","PLS","PLS_std","RF","RF_std","ExtraTrees",
             "ExtraTrees_std","Bagging","Bagging_std"]
    metrics_df=pd.DataFrame(data=metrics,columns=columns)
    return(metrics_df)
# get average predictions for CV
def average_preds_CV(data):
    metrics=[]
    mets=["MLR","ANN","SVM","PLS","RF","ExtraTrees","Bagging"]
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
    columns=["mol_name","Smiles","N","MLR","MLR_std","ANN", "ANN_std","SVM","SVM_std","PLS","PLS_std","RF","RF_std","ExtraTrees",
             "ExtraTrees_std","Bagging","Bagging_std"]
    metrics_df=pd.DataFrame(data=metrics,columns=columns)
    return(metrics_df)
# parameters to change	
# train and test set dataset locations
train_loc="train.csv"
test_loc="test.csv"
train=pd.read_csv(train_loc)
test=pd.read_csv(test_loc)
#full dataset
full_loc="full.csv"
full=pd.read_csv(full_loc)
# save file locations
output="stat_split\\GAS_SET_full.csv"
output_metrics="stat_split\\GAS_SET_full_metrics.csv"
output_CV="CV\\GAS_SET_full.csv"
output_metrics_CV="CV\\GAS_SET_full_metrics.csv"
# descriptors to use in models
descs=["N","N_MW","N_E0","N_G0","N_vol","N_HOMO","N_LUMO","N_DM","EB",
       "EA_H","EA_nonH","N_TCA","N_BAD","sol_PCA1","sol_PCA2","sol_PCA3",
       "sol_PCA4","sol_PCA5","sol_PCA_MP","sol_PCA_BP","sol_PCA_Density",
       "N_fukui","N_fukui_Li","N_fukui_charge"]
# SVM parameters found in grid search
C=19
E=1
G=0.02
# end of parameters to change
# perform fixed split
# define repeats
rep=10
# define master lists for metrics and preds
metrics=[]
preds=[]
# run ML
for f in range(rep):
    metric,pred=stat_split_metrics(train,test,C,E,G,descs)
    metrics.append(metric)
    preds.append(pred)
# get average metrics and predictions from the 10 runs
metrics=average_metrics(metrics)
preds=average_preds(preds)
# save to file
preds.to_csv(output)
metrics.to_csv(output_metrics)
# perform 10-fold CV
# number of repeats
rep=10
preds_CV=[]
for f in range(rep):
    metrics_CV,preds=CV_metrics(full,10,C,E,G,descs)
    preds_CV.append(preds)
# get average predictions from 10 repeats
preds_CV=average_preds_CV(preds_CV)
# save to file
preds_CV.to_csv(output_CV)
# note that the metrics are from the average of a single run of 10-CV NOT from the average predictions
metrics_CV.to_csv(output_metrics_CV)
