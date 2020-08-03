#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 20:03:07 2020
@author: GrupoA
"""

import numpy as np
import pandas as pd


class Perceptron_Dual(object):
    
    def __init__(self, nr_iterations=10, kernel='none', sigma=1.0, theta=0.0, degree=2.0, eta=1.0, c1=0.0, c2=1.0):
        self.nr_iterations = nr_iterations  
        self.kernel = kernel 
        self.sigma = sigma 
        self.theta = theta 
        self.degree = degree 
        self.eta = eta  
        self.c1 = c1 
        self.c2 = c2  

    def fit(self, x, y):
        self.alpha_ = np.zeros(x.shape[0])
        self.errors_ = []
        self.x_ = np.copy(x)
        self.y_ = y
        print('> Degree:',self.degree)
        for iteration in range(self.nr_iterations):
            errors = 0
            print('> Current iteration: %.i' % iteration)
            for j in range(self.alpha_.size):
                yhat = self.predict(self.x_[j, :])
                if yhat != y[j]:
                    self.alpha_[j] += 1
                    errors += 1
            if errors == 0:
                print('> Converged after %.i iterations' % iteration)
                break
        
        return self

    def predict(self, xj):
        total = 0 
        for i in range(self.alpha_.size):
            total += self.alpha_[i] * self.y_[i] * self.kernel_function(self.x_[i, :], xj)
        return np.where(total >= 0.0, 1, -1)

    def kernel_function(self, xi, xj):
        if self.kernel == 'none':
            kern = np.dot(xi, xj)
        elif self.kernel == 'rbf':
            num = np.linalg.norm(xi-xj) ** 2
            den = 2 * self.sigma ** 2
            frac = num / den
            kern = np.exp(-1 * frac)
        elif self.kernel == 'polynomial':
            kern = (np.dot(xi, xj) + self.theta)**self.degree
        else:
            raise ValueError('Unrecognized kernel "' + self.kernel + '"') 
        return self.c1 + self.c2 * kern  


class MCP_Dual(object):

    def __init__(self, num_classifiers, max_iters=200, kernel='none', sigma=1.0, theta=0.0, degree=2.0, eta=1.0, c1=0.0, c2=1.0):
        self.kernel = kernel
        self.degree = degree
        self.sigma = sigma
        self.tot_binary_classifiers = num_classifiers
        self.classifiers = np.empty(num_classifiers, dtype=object) 
        for i in range(int(num_classifiers)):
            self.classifiers[i] = Perceptron_Dual(max_iters, kernel, sigma, theta, degree, eta, c1, c2)  

    def fit(self, X, y):
        for i in range(int(self.tot_binary_classifiers)):            
            print('-> Start of Training Binary Classifier (OVA): %s/%s' % (str(i+1), str(self.tot_binary_classifiers)))
            self.classifiers[i].fit(X, y)
            print('-> End of Training Binary Classifier (OVA): %s/%s' % (str(i+1), str(self.tot_binary_classifiers)))
            print(45*'-')
            