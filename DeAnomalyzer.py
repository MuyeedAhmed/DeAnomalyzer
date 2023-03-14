#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 15:08:33 2023

@author: muyeedahmed
"""
import sys
import subprocess
import os
import pandas as pd
import mat73
from scipy.io import loadmat
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
from sklearn import metrics
from copy import copy, deepcopy
from sklearn.metrics.cluster import adjusted_rand_score
import bisect 
from Paramaters import getParameter
import matlab.engine
            
datasetFolderDir = 'Dataset/'
fname = str(sys.argv[1])

class DeAnomalyzer:
    def __init__(self, algoName, tool, fileName):
        self.algoName = algoName
        self.tool = tool
        self.fileName = fileName
        self.parameters = getParameter(algoName, tool)
        self.withGT = False
        self.X = []
        self.y = []
        
        self.readData()
        self.injectParameter()
        
        if self.tool == "Matlab":
            self.eng = matlab.engine.start_matlab()
    
    def destroy(self):
        if self.tool == "Matlab":
            self.eng.quit()
            
    def injectParameter(self):
        if self.algoName == "IF" and self.tool == "Sklearn":
            auto_1 = min(256, self.X.shape[0])/self.X.shape[0]
            bisect.insort(self.parameters[1][2], auto_1)
            self.parameters[1][2][self.parameters[1][2].index(auto_1)] = 'auto'
        elif self.algoName == "IF" and self.tool == "Matlab":
            if_cont, lof_cont = self.IF_LOF_ContFactor(self.X)
            if lof_cont != 0:
                bisect.insort(self.parameters[0][2], lof_cont)
            if if_cont != 0:
                bisect.insort(self.parameters[0][2], if_cont)
            
            auto_1 = min(256, self.X.shape[0])/self.X.shape[0]
            bisect.insort(self.parameters[2][2], auto_1)
            self.parameters[2][2][self.parameters[2][2].index(auto_1)] = 'auto'
        elif self.algoName == "EE" and self.tool == "Sklearn":
            if_cont, lof_cont = self.IF_LOF_ContFactor(self.X)
            if lof_cont != 0:
                bisect.insort(self.parameters[3][2], lof_cont)
            if if_cont != 0:
                bisect.insort(self.parameters[3][2], if_cont)
        elif self.algoName == "OCSVM" and self.tool == "Matlab":
            if_cont, lof_cont = self.IF_LOF_ContFactor(self.X)
            if lof_cont != 0:
                bisect.insort(self.parameters[0][2], lof_cont)
            if if_cont != 0:
                bisect.insort(self.parameters[0][2], if_cont)
      
    def readData(self):
        folderpath = datasetFolderDir
        if os.path.exists(folderpath+self.fileName+".mat") == 1:
            try:
                df = loadmat(folderpath+self.fileName+".mat")
            except NotImplementedError:
                df = mat73.loadmat(folderpath+self.fileName+".mat")
        
            gt=df["y"]
            self.y = gt.reshape((len(gt)))
            self.X=df['X']
            
            if np.isnan(self.X).any():
                print("Error: File contains NaN")
                return
        elif os.path.exists(folderpath+self.fileName+".csv") == 1:
            df = pd.read_csv(folderpath+self.fileName+".csv")
            self.X = df.drop("target", axis=1)
            if 'target' in df.columns:
                target = df["target"].to_numpy()
                self.y = target
                self.withGT = True
            else:
                gt = []
                self.withGT = False
            if self.X.isna().any().any() == 1:
                print("Error: File contains NaN")
                return
        else:
            print("Error: File doesn't exist")
            return
    
    def IF_LOF_ContFactor(self):
        labels = []
        num_label = 0
        for i in range(5):
            clustering = IsolationForest().fit(self.X)
        
            l = clustering.predict(self.X)
            num_label = len(l)
            l = [0 if x == 1 else 1 for x in l]
            labels.append(l)
        _, counts_if = np.unique(labels, return_counts=True)
        if_per = min(counts_if)/(num_label*5)
    
        
        labels_lof = LocalOutlierFactor().fit_predict(self.X)
        _, counts_lof = np.unique(labels_lof, return_counts=True)
        lof_per = min(counts_lof)/(num_label)
        
        if if_per == 1:
            if_per = 0
        if lof_per == 1:
            lof_per = 0
        
        return if_per, lof_per 
    
    def get_blind_route(self):
        blind_route = []
        
        for p_i in range(len(self.parameters)):
            p = p_i
            
            parameter_route = []
            ari_scores = []
            passing_param = deepcopy(self.parameters)
    
            default_f1, default_ari = self.runAlgo(passing_param)
    
            parameter_route.append([passing_param[p][1], default_ari, default_f1])
            ari_scores.append(default_ari)
            
            i_def = passing_param[p][2].index(passing_param[p][1])
            if i_def+1 == len(self.parameters[p][2]):
                i_pv = i_def-1    
            else:
                i_pv = i_def+1
            
            while True:
                if i_pv >= len(self.parameters[p][2]):
                    break
                if i_pv < 0:
                    break
    
                passing_param[p][1] = self.parameters[p][2][i_pv]
                f1_score, ari_score = self.runAlgo(passing_param)
    
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
            self.parameters[p][1] = parameter_route[max_index][0]
            blind_route.append([self.parameters[p][0], max_index, default_index, parameter_route])
        return blind_route
        
    def get_guided_route(self):
        guided_route = []
        
        for p_i in range(len(self.parameters)):
            p = p_i
            
            parameter_route = []
            ari_scores = []
            f1_scores = []
            passing_param = deepcopy(self.parameters)
    
            default_f1, default_ari = self.runAlgo(passing_param)
    
            parameter_route.append([passing_param[p][1], default_ari, default_f1])
            ari_scores.append(default_ari)
            f1_scores.append(default_f1)
    
            i_def = passing_param[p][2].index(passing_param[p][1])
            if i_def+1 == len(self.parameters[p][2]):
                i_pv = i_def-1    
            else:
                i_pv = i_def+1
            
            while True:
                if i_pv >= len(self.parameters[p][2]):
                    break
                if i_pv < 0:
                    break
    
                passing_param[p][1] = self.parameters[p][2][i_pv]
                f1_score, ari_score = self.runAlgo(passing_param)
    
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
            self.parameters[p][1] = parameter_route[max_index][0]
            guided_route.append([self.parameters[p][0], max_index, default_index, parameter_route])
        return guided_route

    def runAlgo(self, params):
        
        
        labelFile = self.fileName + "_" + str(params[0][1]) + "_" + str(params[1][1]) + "_" + str(params[2][1]) + "_" + str(params[3][1]) + "_" + str(params[4][1]) + "_" + str(params[5][1]) + "_" + str(params[6][1]) + "_" + str(params[7][1])
        
        if os.path.exists("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile+".csv") == 0:
            frr=open("GD_ReRun/MatOCSVM.csv", "a")
            frr.write(self.fileName+","+str(params[0][1])+","+str(params[1][1])+","+str(params[2][1])+","+str(params[3][1])+","+str(params[4][1])+","+str(params[5][1])+","+str(params[6][1])+","+str(params[7][1])+'\n')
            frr.close()
            try:
                self.eng.MatOCSVM_Rerun(nargout=0)
                frr=open("GD_ReRun/MatOCSVM.csv", "w")
                frr.write('Filename,ContaminationFraction,KernelScale,Lambda,NumExpansionDimensions,StandardizeData,BetaTolerance,BetaTolerance,GradientTolerance,IterationLimit\n')
                frr.close()
                if os.path.exists("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile+".csv") == 0:      
                    print("\nFaild to run Matlab Engine from Python.\n")
                    exit(0)
            except:
                print("\nFaild to run Matlab Engine from Python.\n")
                exit(0)    
        
        
        labels =  pd.read_csv("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile+".csv", header=None).to_numpy()
        


        f1 = []
        ari = []
        if self.withGT:
            for i in range(10):
                f1.append(metrics.f1_score(self.y, labels[i]))
            
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
              ari.append(adjusted_rand_score(labels[i], labels[j]))
              
        if self.withGT:
            return np.mean(f1), np.mean(ari)
        else:
            return -1, np.mean(ari) 

        
if __name__ == '__main__':
    print("DeAnomalyzer\n\n")
    
    da = DeAnomalyzer("IF", "Sklearn", "ar1")
    
    blind_route = da.get_blind_route()
    
    DefaultARI = str(blind_route[0][3][0][1])
    DefaultF1 = str(blind_route[0][3][0][2])
    print("Default settings: ")
    print("\tCross-run ARI: ", DefaultARI)
    if da.withGT:
        print("\tF1 Score: ", DefaultF1)
    
    UninformedARI = str(blind_route[-1][3][-1][1])
    UninformedF1 = str(blind_route[-1][3][-1][2])
    print("Univariate Search: ")
    print("\tCross-run ARI: ", UninformedARI)
    if da.withGT:
        print("\tF1 Score: ", UninformedF1)
    print("\tOutput Parameters:")
    print("\n\t", end='')
    for i in range(len(blind_route)):
        print(blind_route[i][0],":", blind_route[i][3][blind_route[i][1]][0], end=', ')
    print("\n")
    
    if da.withGT:
        guided_route = da.get_guided_route()
    
        InformedARI = str(guided_route[-1][3][-1][1])
        InformedF1 = str(guided_route[-1][3][-1][2])
        print("Bivariate Search: ")
        print("\tCross-run ARI: ", InformedARI)
        print("\tF1 Score: ", InformedF1)
        print("\tOutput Parameters:")
        print("\n\t", end='')
        for i in range(len(guided_route)):
            print(guided_route[i][0],":", guided_route[i][3][guided_route[i][1]][0], end=', ')
        print("\n")
    
    da.destroy()
    
    
    # print("Select one of the following action by entering the serial number")
    # print("\t1. Reduce Non-determinism in Isolation Forest: Scikit-Learn")
    # print("\t2. Reduce Non-determinism in Isolation Forest: Matlab")
    # print("\t3. Reduce Non-determinism in Isolation Forest: R")
    # print("\t4. Reduce Non-determinism in Robust Covariance: Scikit-Learn")
    # print("\t5. Reduce Non-determinism in Robust Covariance: Matlab")
    # print("\t6. Reduce Non-determinism in One Class SVM: Matlab")
    # print()
    # print("\t7. Reduce Inconsistency in Isolation Forest")
    # print("\t8. Reduce Inconsistency in Robust Covariance")
    # print("\t9. Reduce Inconsistency in Local Outlier Factor")
    # print("\t10. Reduce Inconsistency in One Class SVM")
    # print("Press Enter after completion")
    
    # n = int(input("Choice: "))
    # if n==1:
    #     subprocess.Popen(["python", "SkIF_GD.py", fname])
    # elif n ==2:
    #     subprocess.Popen(["python", "MatIF_GD.py", fname])
    # elif n ==3:
    #     subprocess.Popen(["python", "RIF_GD.py", fname])
    # elif n ==4:
    #     subprocess.Popen(["python", "SkEE_GD.py", fname])
    # elif n ==5:
    #     subprocess.Popen(["python", "MatEE_GD.py", fname])
    # elif n ==6:
    #     subprocess.Popen(["python", "MatOCSVM_GD.py", fname])
        
    # elif n ==7:
    #     subprocess.Popen(["python", "Inconsistency_IF.py", fname])
    # elif n ==8:
    #     subprocess.Popen(["python", "Inconsistency_EE.py", fname])
    # elif n ==9:
    #     subprocess.Popen(["python", "Inconsistency_LOF.py", fname])
    # elif n ==10:
    #     subprocess.Popen(["python", "Inconsistency_OCSVM.py", fname])
    # else:
    #     print("Try Again")
        
    