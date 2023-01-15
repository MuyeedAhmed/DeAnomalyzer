import warnings
warnings.filterwarnings("ignore")
from sklearn.covariance import EllipticEnvelope
import sys
import glob
import os
import pandas as pd
import mat73
from scipy.io import loadmat
import numpy as np
from sklearn import metrics
from copy import deepcopy
from sklearn.metrics.cluster import adjusted_rand_score
import bisect
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


datasetFolderDir = 'Dataset/'
withGT = True

def ee(filename, parameters, parameter_iteration):
    print(filename)
    folderpath = datasetFolderDir
    parameters_this_file = deepcopy(parameters)
    global withGT
    
    if os.path.exists(folderpath+filename+".mat") == 1:
        try:
            df = loadmat(folderpath+filename+".mat")
        except NotImplementedError:
            df = mat73.loadmat(folderpath+filename+".mat")

        gt=df["y"]
        gt = gt.reshape((len(gt)))
        X=df['X']
        if np.isnan(X).any():
            print("File contains NaN")
            return
    elif os.path.exists(folderpath+filename+".csv") == 1:
        X = pd.read_csv(folderpath+filename+".csv")
        if 'target' in X.columns:
            target=X["target"].to_numpy()
            X=X.drop("target", axis=1)
            gt = target
        else:
            gt = []
            withGT = False
        if X.isna().any().any() == 1:
            print("File contains NaN")
            return
    else:
        print("File doesn't exist")
        return
    
    # ## Rearrange "IF" and "LOF"
    if_cont, lof_cont = IF_LOF_ContFactor(X)
    mod_parameters = deepcopy(parameters)    
    if lof_cont != 0:
        bisect.insort(mod_parameters[3][2], lof_cont)
    if if_cont != 0:
        bisect.insort(mod_parameters[3][2], if_cont)
    # ##
    
    blind_route = get_blind_route(X, gt, filename, deepcopy(mod_parameters), parameter_iteration)
    
    DefaultARI = str(blind_route[0][3][0][1])
    DefaultF1 = str(blind_route[0][3][0][2])
    print("Default settings: ")
    print("\tCross-run ARI: ", DefaultARI)
    if withGT:
        print("\tF1 Score: ", DefaultF1)
    
    UninformedARI = str(blind_route[-1][3][-1][1])
    UninformedF1 = str(blind_route[-1][3][-1][2])
    print("Univariate Search: ")
    print("\tCross-run ARI: ", UninformedARI)
    if withGT:
        print("\tF1 Score: ", UninformedF1)

    
    frr=open("Results/SkEE_Uni.csv", "a")
    frr.write(filename)
    for i in range(len(blind_route)):
        frr.write("," + str(blind_route[i][3][blind_route[i][1]][0]))
    frr.write("\n")
    frr.close()
    
    if withGT:
        guided_route = get_guided_route(X, gt, filename, deepcopy(mod_parameters), parameter_iteration)
    
        InformedARI = str(guided_route[-1][3][-1][1])
        InformedF1 = str(guided_route[-1][3][-1][2])
        print("Bivariate Search: ")
        print("\tCross-run ARI: ", InformedARI)
        print("\tF1 Score: ", InformedF1)

        frr=open("Results/SkEE_Bi.csv", "a")
        frr.write(filename)
        for i in range(len(guided_route)):
            frr.write("," + str(guided_route[i][3][guided_route[i][1]][0]))
        frr.write("\n")
        frr.close()
        
def IF_LOF_ContFactor(X):
    labels = []
    num_label = 0
    for i in range(5):
        clustering = IsolationForest().fit(X)
    
        l = clustering.predict(X)
        num_label = len(l)
        l = [0 if x == 1 else 1 for x in l]
        labels.append(l)
    _, counts_if = np.unique(labels, return_counts=True)
    if_per = min(counts_if)/(num_label*5)

    
    labels_lof = LocalOutlierFactor().fit_predict(X)
    _, counts_lof = np.unique(labels_lof, return_counts=True)
    lof_per = min(counts_lof)/(num_label)
    
    if if_per == 1:
        if_per = 0
    if lof_per == 1:
        lof_per = 0
    
    return if_per, lof_per
    
def get_blind_route(X, gt, filename, parameters_this_file, parameter_iteration):
    blind_route = []
    
    for p_i in range(len(parameters_this_file)):
        p = p_i
        
        parameter_route = []
        ari_scores = []
        passing_param = deepcopy(parameters_this_file)

        default_f1, default_ari = runEE(filename, X, gt, passing_param, parameter_iteration)

        parameter_route.append([passing_param[p][1], default_ari, default_f1])
        ari_scores.append(default_ari)
        
        i_def = passing_param[p][2].index(passing_param[p][1])
        if i_def+1 == len(parameters_this_file[p][2]):
            i_pv = i_def-1    
        else:
            i_pv = i_def+1
        
        while True:
            if i_pv >= len(parameters_this_file[p][2]):
                break
            if i_pv < 0:
                break

            passing_param[p][1] = parameters_this_file[p][2][i_pv]
            f1_score, ari_score = runEE(filename, X, gt, passing_param, parameter_iteration)

            if ari_score > np.max(ari_scores):
                parameter_route.append([passing_param[p][1], ari_score, f1_score])
                ari_scores.append(ari_score)
            
            if ari_score != np.max(ari_scores):
                
                if i_pv - 1 > i_def:
                    break
                elif i_pv - 1 == i_def:
                    i_pv = i_def - 1
                else:
                    break
            else:
                if i_pv > i_def:
                    i_pv += 1
                else:
                    i_pv -= 1
        
        max_index = ari_scores.index(max(ari_scores))
        default_index = ari_scores.index(default_ari)
        parameters_this_file[p][1] = parameter_route[max_index][0]
        blind_route.append([parameters_this_file[p][0], max_index, default_index, parameter_route])
    return blind_route
    
def get_guided_route(X, gt, filename, parameters_this_file, parameter_iteration):
    guided_route = []
    
    for p_i in range(len(parameters_this_file)):
        p = p_i
        
        parameter_route = []
        ari_scores = []
        f1_scores = []
        passing_param = deepcopy(parameters_this_file)

        default_f1, default_ari = runEE(filename, X, gt, passing_param, parameter_iteration)

        parameter_route.append([passing_param[p][1], default_ari, default_f1])
        ari_scores.append(default_ari)
        f1_scores.append(default_f1)

        i_def = passing_param[p][2].index(passing_param[p][1])
        if i_def+1 == len(parameters_this_file[p][2]):
            i_pv = i_def-1    
        else:
            i_pv = i_def+1
        
        while True:
            if i_pv >= len(parameters_this_file[p][2]):
                break
            if i_pv < 0:
                break

            passing_param[p][1] = parameters_this_file[p][2][i_pv]
            f1_score, ari_score = runEE(filename, X, gt, passing_param, parameter_iteration)

            if ari_score > np.max(ari_scores) and f1_score > np.max(f1_scores):
                parameter_route.append([passing_param[p][1], ari_score, f1_score])
                ari_scores.append(ari_score)
                f1_scores.append(f1_score)            
            if ari_score != np.max(ari_scores) and f1_score != np.max(f1_scores):
                
                if i_pv - 1 > i_def:
                    break
                elif i_pv - 1 == i_def:
                    i_pv = i_def - 1
                else:
                    break
            else:
                if i_pv > i_def:
                    i_pv += 1
                else:
                    i_pv -= 1
        max_index = ari_scores.index(max(ari_scores))
        default_index = ari_scores.index(default_ari)
        parameters_this_file[p][1] = parameter_route[max_index][0]
        guided_route.append([parameters_this_file[p][0], max_index, default_index, parameter_route])
    return guided_route  
    


def runEE(filename, X, gt, params, parameter_iteration):
    labelFile = filename + "_" + str(params[0][1]) + "_" + str(params[1][1]) + "_" + str(params[2][1]) + "_" + str(params[3][1])
    global withGT
    sp = params[0][1]
    ac = params[1][1]
    sf = params[2][1]
    cont = params[3][1]
    
    labels = []
    f1 = []
    ari = []
    for i in range(10):
        clustering = EllipticEnvelope(store_precision=sp, assume_centered=ac, 
                                     support_fraction=sf, contamination=cont).fit(X)
    
        l = clustering.predict(X)
        l = [0 if x == 1 else 1 for x in l]
        labels.append(l)
        if withGT:
            f1.append(metrics.f1_score(gt, l))
    
    if os.path.exists("Labels/EE_Sk/Labels_Sk_EE_"+labelFile+".csv") == 0:
        fileLabels=open("Labels/EE_Sk/Labels_Sk_EE_"+labelFile+".csv", 'a')
        for l in labels:
            fileLabels.write(','.join(str(s) for s in l) + '\n')
        fileLabels.close()    
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
          ari.append(adjusted_rand_score(labels[i], labels[j]))      
    

    if withGT:
        return np.mean(f1), np.mean(ari)
    else:
        return -1, np.mean(ari) 
   
    
    
if __name__ == '__main__':
    folderpath = datasetFolderDir
    master_files1 = glob.glob(folderpath+"*.mat")
    master_files2 = glob.glob(folderpath+"*.csv")
    master_files = master_files1 + master_files2
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    if os.path.exists("Results/SkEE_Uni.csv"):
        SkEE_Uni = pd.read_csv("Results/SkEE_Uni.csv")
        done_files = SkEE_Uni["Filename"].to_numpy()
        master_files = [item for item in master_files if item not in done_files]
    master_files.sort()
        
    parameters = []
    store_precision = [True, False]
    assume_centered = [True, False]
    support_fraction = [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    contamination = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    parameters.append(["store_precision", True, store_precision])
    parameters.append(["assume_centered", False, assume_centered])
    parameters.append(["support_fraction", None, support_fraction])
    parameters.append(["contamination", 0.1, contamination])
    if os.path.exists("Results/SkEE_Uni.csv") == 0:
        frr=open("Results/SkEE_Uni.csv", "w")
        frr.write('Filename,store_precision,assume_centered,support_fraction,contamination\n')
        frr.close()
    if os.path.exists("Results/SkEE_Bi.csv") == 0:
        frr=open("Results/SkEE_Bi.csv", "w")
        frr.write('Filename,store_precision,assume_centered,support_fraction,contamination\n')
        frr.close()
    
    for fname in master_files:
        ee(fname, parameters, 0)
        
        
        