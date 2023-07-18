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
            

class DeAnomalyzer:
    def __init__(self, algoName, tool):
        self.algoName = algoName
        self.tool = tool
        self.parameters = getParameter(algoName, tool)
        self.withGT = False
        self.X = []
        self.y = []
        
        
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
      
    def readData(self, datapath):
        if os.path.exists(datapath):
            if datapath[-3:] == "mat":
                try:
                    df = loadmat(datapath)
                except NotImplementedError:
                    df = mat73.loadmat(datapath)
        
                gt=df["y"]
                self.y = gt.reshape((len(gt)))
                self.X=df['X']
                
                if np.isnan(self.X).any():
                    print("Error: File contains NaN")
                    return
            elif datapath[-3:] == "csv":
                df = pd.read_csv(datapath)
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
                print("Error: File needs to be either .mat or .csv")
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

    def fit(self, datapath, mode="all"):
        self.fileName = datapath.split("/")[-1][:-4]
        self.readData(datapath)
        self.injectParameter()

        if mode == "all":
            blind_route = self.get_blind_route()
            
            self.default_ari = str(blind_route[0][3][0][1])
            self.univariate_ari = str(blind_route[-1][3][-1][1])
            
            print("\tOutput Parameters:")
            print("\n\t", end='')
            for i in range(len(blind_route)):
                print(blind_route[i][0],":", blind_route[i][3][blind_route[i][1]][0], end=', ')
            print("\n")
            
            if self.withGT:
                self.default_f1 = str(blind_route[0][3][0][2])
                self.univariate_f1 = str(blind_route[-1][3][-1][2])
                
                guided_route = self.get_guided_route()
            
                self.bivariate_ari = str(guided_route[-1][3][-1][1])
                self.bivariate_f1 = str(guided_route[-1][3][-1][2])
                
                print("\tOutput Parameters:")
                print("\n\t", end='')
                for i in range(len(guided_route)):
                    print(guided_route[i][0],":", guided_route[i][3][guided_route[i][1]][0], end=', ')
                print("\n")
            
            self.destroy()
            return self
            
