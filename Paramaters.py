#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 15:57:06 2023

@author: muyeedahmed
"""

# class Parameters:
#     # def __init__(self, algoName, tool):
#     #     self.algoName = algoName
#     #     self.tool = tool
def getParameter(algoName, tool="Inconsistency"):
    parameters = []
    
    if algoName == "IF" and tool == "Sklearn":
        n_estimators = [2, 4, 8, 16, 32, 64, 100, 128, 256, 512]##
        max_samples = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        contamination = ['auto'] 
        max_features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        bootstrap = [True, False]
        n_jobs = [1, None] 
        warm_start = [True, False]
        
        parameters.append(["n_estimators", 100, n_estimators])
        parameters.append(["max_samples", 'auto', max_samples])
        parameters.append(["contamination", 'auto', contamination])
        parameters.append(["max_features", 1.0, max_features])
        parameters.append(["bootstrap", False, bootstrap])
        parameters.append(["n_jobs", None, n_jobs])
        parameters.append(["warm_start", False, warm_start])
    
    elif algoName == "IF" and tool == "Matlab":
        ContaminationFraction = [0, 0.05, 0.1, 0.15, 0.2, 0.25];
        NumLearners = [1, 2, 4, 8, 16, 32, 64, 100, 128, 256, 512];
        NumObservationsPerLearner = [0.05, 0.1, 0.2, 0.5, 1];
        
        parameters.append(["ContaminationFraction", 0.1, ContaminationFraction])
        parameters.append(["NumLearners", 100, NumLearners])
        parameters.append(["NumObservationsPerLearner", 'auto', NumObservationsPerLearner])
    
    elif algoName == "IF" and tool == "R":
        ntrees = [2, 4, 8, 16, 32, 64, 100, 128, 256, 512]
        standardize_data = ["TRUE","FALSE"]
        sample_size = ['auto',0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,"NULL"]
        ncols_per_tree = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,'def']
        
        parameters.append(["ntrees",512,ntrees])
        parameters.append(["standardize_data","TRUE",standardize_data])
        parameters.append(["sample_size",'auto',sample_size])
        parameters.append(["ncols_per_tree",'def',ncols_per_tree])
    
    elif algoName == "EE" and tool == "Sklearn":
        store_precision = [True, False]
        assume_centered = [True, False]
        support_fraction = [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        contamination = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        parameters.append(["store_precision", True, store_precision])
        parameters.append(["assume_centered", False, assume_centered])
        parameters.append(["support_fraction", None, support_fraction])
        parameters.append(["contamination", 0.1, contamination])
    
    elif algoName == "EE" and tool == "Matlab":
        Method = ["olivehawkins", "fmcd", "ogk"];
        OutlierFraction = [0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5];
        NumTrials = [2, 5, 10, 25, 50, 100, 250, 500, 750, 1000];
        BiasCorrection = [1, 0];
        NumOGKIterations = [1, 2, 3];
        UnivariateEstimator = ["tauscale", "qn"];
        ReweightingMethod = ["rfch", "rmvn"];
        NumConcentrationSteps = [2, 5, 10, 15, 20];
        StartMethod = ["elemental","classical", "medianball"];    
        
        parameters.append(["Method", "fmcd", Method])
        parameters.append(["OutlierFraction", 0.5, OutlierFraction])
        parameters.append(["NumTrials", 500, NumTrials])
        parameters.append(["BiasCorrection", 1, BiasCorrection])
        parameters.append(["NumOGKIterations", 2, NumOGKIterations])
        parameters.append(["UnivariateEstimator", "tauscale", UnivariateEstimator])
        parameters.append(["ReweightingMethod", "rfch", ReweightingMethod])
        parameters.append(["NumConcentrationSteps", 10, NumConcentrationSteps])
        parameters.append(["StartMethod", "classical", StartMethod])
        
    # elif algoName == "LOF" and tool == "Sklearn":
    #     print()
    # elif algoName == "LOF" and tool == "Matlab":
    #     print()
    # elif algoName == "LOF" and tool == "R":
    #     print()
    elif algoName == "OCSVM" and tool == "Sklearn":
        kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
        degree = [3, 4, 5, 6] # Kernel poly only
        gamma = ['scale', 'auto'] # Kernel ‘rbf’, ‘poly’ and ‘sigmoid’
        coef0 = [0.0, 0.1, 0.2, 0.3, 0.4] # Kernel ‘poly’ and ‘sigmoid’
        tol = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        nu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        shrinking = [True, False]
        cache_size = [50, 100, 200, 400]
        max_iter = [50, 100, 150, 200, 250, 300, -1]
        
        parameters.append(["kernel", 'rbf', kernel])
        parameters.append(["degree", 3, degree])
        parameters.append(["gamma", 'scale', gamma])
        parameters.append(["coef0", 0.0, coef0])
        parameters.append(["tol", 0.001, tol])
        parameters.append(["nu", 0.5, nu])
        parameters.append(["shrinking", True, shrinking])
        parameters.append(["cache_size", 200, cache_size])
        parameters.append(["max_iter", -1, max_iter])
        
    elif algoName == "OCSVM" and tool == "Matlab":
        ContaminationFraction = [0.05, 0.1, 0.15, 0.2, 0.25];
        KernelScale = [1, "auto", 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5];
        Lambda = ["auto", 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5];
        NumExpansionDimensions = ["auto", 2^12, 2^15, 2^17, 2^19];
        StandardizeData = [0, 1];
        BetaTolerance = [1e-2, 1e-3, 1e-4, 1e-5];
        GradientTolerance = [1e-3, 1e-4, 1e-5, 1e-6];
        IterationLimit = [100, 200, 500, 1000, 2000];
        
        parameters.append(["ContaminationFraction", 0.1, ContaminationFraction])
        parameters.append(["KernelScale", 1, KernelScale])
        parameters.append(["Lambda", 'auto', Lambda])
        parameters.append(["NumExpansionDimensions", 'auto', NumExpansionDimensions])
        parameters.append(["StandardizeData", 0, StandardizeData])
        parameters.append(["BetaTolerance", 1e-4, BetaTolerance])
        parameters.append(["GradientTolerance", 1e-4, GradientTolerance])
        parameters.append(["IterationLimit", 1000, IterationLimit])
    
    elif algoName == "OCSVM" and tool == "R":
        kernel = ['linear', 'polynomial', 'radial', 'sigmoid']
        degree = [3, 4, 5, 6]
        gamma = ['scale', 'auto']
        coef0 = [0, 0.1, 0.2, 0.3, 0.4]
        tolerance = [0.1, 0.01, 0.001, 0.0001]
        nu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        shrinking = ["TRUE", "FALSE"]
        cachesize = [50, 100, 200, 400]
        epsilon = [0.1, 0.2, 0.01, 0.05]
        
        parameters.append(["kernel", 'radial', kernel])
        parameters.append(["degree", 3, degree])
        parameters.append(["gamma","scale",gamma])
        parameters.append(["coef0", 0, coef0])
        parameters.append(["tolerance", 0.001, tolerance])
        parameters.append(["nu", 0.5, nu])
        parameters.append(["shrinking", "TRUE", shrinking])
        parameters.append(["cachesize", 200, cachesize])
        parameters.append(["epsilon",0.1,epsilon])
    
    return parameters
    
    