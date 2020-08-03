#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 20:03:07 2020

@author: GrupoA
"""

import numpy as np

class Perceptron_Primal(object):
   
    def __init__(self, learningrate, max_iterations):
        self.learningrate = learningrate
        self.max_iterations = max_iterations
        
    def fit(self, X, y):
        iterations = 0
        mismatch_flag = 1
        #self.w = np.array([0] * len(X[0]))
        while iterations <= self.max_iterations and mismatch_flag != 0:
            mismatch_flag = 0
            self.w = np.array([0] * len(X[0]))
            for i in range(len(X)):
                y_hat = np.dot(self.w.T, X[i,:])    
                if y_hat >= 0.0:
                    y_hat = 1
                else:
                    y_hat = -1
                if y[i]*y_hat <= 0:
                    mismatch_flag = 1
                    self.w = self.w + self.learningrate * y[i] * X[i]
            iterations += 1
        print('> Max iterations:', iterations-1)
 
    def predict(self, X):
        final_scores = np.array([np.dot(self.w.T, x) for x in X])
        preds = [1 if x >= 0.0 else -1 for x in final_scores]
        return preds

class MCP_Primal(object):

    def __init__(self, num_classifiers, max_iters=200, learningrate=0.1):
        self.tot_binary_classifiers = num_classifiers
        self.classifiers = np.empty(num_classifiers, dtype=object) 
        for i in range(int(num_classifiers)):
            self.classifiers[i] = Perceptron_Primal(learningrate, max_iters)  
        
    def fit(self, X, y):
        for i in range(int(self.tot_binary_classifiers)):
            print('-> Start of Training Binary Classifier (OVA): %s/%s' % (str(i+1), str(self.tot_binary_classifiers)))
            self.classifiers[i].fit(X, y)
            print('-> End of Training Binary Classifier (OVA): %s/%s' % (str(i+1), str(self.tot_binary_classifiers)))
            print(45*'-')
        