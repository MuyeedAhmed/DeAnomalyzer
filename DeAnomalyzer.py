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
import numpy as np
from sklearn import metrics
from copy import copy, deepcopy
from sklearn.metrics.cluster import adjusted_rand_score
import bisect 
from Paramaters import getParameter

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

    def runAlgo(self):
        print("Incomplete")
        
        
if __name__ == '__main__':
    
    da = DeAnomalyzer("IF", "Sklearn", "ar1")
    
    # print("DeAnomalyzer\n\n")
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
        
    