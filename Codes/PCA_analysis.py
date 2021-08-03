# This python script creates figures for PCA analysis (saves plots and displays them in Jupyter Notebook). Plot 1 creates the cumulative variance captured by principal components (scree plot). Plot 2 plots the first two principal components and colours by N. Plot 3 plots the loading of the descriptors in the first 2 principal components.
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
# parameters to change
# location and name of dataset
full=pd.read_csv("dataset.csv")
# descriptors to be analysed (shown here for GAS_SET)
descs=["sol_PCA1","sol_PCA2","sol_PCA3","sol_PCA4","sol_PCA5","sol_PCA_MP","sol_PCA_BP","sol_PCA_Density","N_MW","N_E0","N_G0","N_vol", "N_HOMO","N_LUMO","N_DM","N_EB","N_EA_H","N_EA_nonH","N_TCA","N_BAD","N_fukui","N_fukui_Li","N_fukui_charge"]
full_desc=full[descs]
# end of parameters to change
# Part 1: Principal Component Analysis
# PCA method
def princ(Data):
    # scale data
    scaler = preprocessing.StandardScaler().fit(Data)
    Data = scaler.transform(Data)
    # set up PCA with n_comp=n_desc
    pca = PCA(n_components=len(descs))
    # get components
    principalComponents = pca.fit_transform(Data)
    cols=[]
    for f in range(len(descs)):
        cols.append('principal component ' + str(f+1))
    principalDf = pd.DataFrame(data = principalComponents, columns = cols)
    # get scree plot data
    cum_scree=np.cumsum(pca.explained_variance_ratio_)*100
    output=[]
    output.append(principalDf)
    output.append(cum_scree)
    return(output)
full_desc_PCA,full_scree=princ(full_desc)

# Plot 1: scree plot
plt.figure(figsize=(4,4))
def getBar(Data,fig_num,title):
    plt.bar(np.arange(len(descs))+1,Data)
    plt.xticks(np.arange(len(descs))+1)
    plt.yticks(np.arange(0,110,10))
    plt.xlim(0.5,len(descs)+ 0.5)
    plt.ylim(0,100)
    plt.title("Cumulative Variance Explained by\nPrincipal Components: " + title,fontsize=11)
    plt.xlabel("Component Number",fontsize=11)
    plt.ylabel("Cumulative Variance (%)",fontsize=11)
    plt.tick_params(labelsize=9)
getBar(full_scree,1,"GAS_SET")
plt.tight_layout()
plt.savefig("gas_scree.png",dpi=600)
plt.show()

# Plot 2: 2D PCA plot coloured by N
plt.figure(figsize=(6,6))
def getPCA_plot(Data,colour,title,fig_num):
    #plt.subplot(3, 2, fig_num)
    plt.scatter(Data['principal component 1'],Data['principal component 2'],c=colour['N'],cmap="brg",s=10)
    plt.title("First Two Principal Components:\nColoured by N - " + title,fontsize=14)
    plt.xlabel("Principal Component 1",fontsize=12)
    plt.ylabel("Principal Component 2",fontsize=12)
    plt.colorbar()
    plt.tight_layout()
    plt.tick_params(labelsize=10)
getPCA_plot(full_desc_PCA,full,"GAS_SET",1)
plt.savefig("gas_PCA.png",dpi=600)
plt.show()

# Plot 3: PCA loadings plot
# marker styles for the descriptors on the plot
mks=[".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D",".","1","2"]
def princ_load(Data,title):
    # scale data
    scaler = preprocessing.StandardScaler().fit(Data)
    Data = scaler.transform(Data)
    # set up PCA with n_comp=n_desc
    pca = PCA(n_components=len(descs)
    Data_pca = pca.fit_transform(Data)
    # get loadings
    loadings = pd.DataFrame(pca.components_.T, columns=[str(x) for x in np.arange(1,len(descs)+1,1)], index=descs)
    fig,ax=plt.subplots(figsize=(8,6))
    # sort colours and markers for descriptors
    NUM_COLORS = len(descs)
    cm = plt.get_cmap('gist_rainbow')
    ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    for f in range(len(descs)):
        ax.scatter(loadings["1"][f],loadings["2"][f], label=descs[f],s=150,alpha=0.6,marker=mks[f])
    plt.legend(bbox_to_anchor=(1, 1),frameon=False)
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.title("Descriptor Loadings Plot:\n" + title)
    plt.tight_layout()
princ_load(full_desc,"GAS_SET")
plt.savefig("gas_loadings.png",dpi=600)
plt.show()
