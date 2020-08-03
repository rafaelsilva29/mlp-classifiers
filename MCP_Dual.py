#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 20:03:07 2020

@author: GrupoA - OML - 19/20
"""

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

np.random.seed(0)

class Perceptron_Dual(object):
    
    def __init__(self, nr_iterations=10, kernel=None, sigma=1.0, theta=0.0, degree=2.0, eta=1.0, c1=0.0, c2=1.0):
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
        aux_it = 0
        for iteration in range(self.nr_iterations):
            errors = 0
            for j in range(self.alpha_.size):
                yhat = self.predict(self.x_[j, :])
                if yhat != y[j]:
                    self.alpha_[j] += 1
                    errors += 1
            self.errors_.append(errors)
            if errors == 0:
                aux_it = iteration
                break
        self.iterations = aux_it

    def predict(self, xj):
        total = 0
        for i in range(self.alpha_.size):
            total += self.alpha_[i] * self.y_[i] * self.kernel_function(self.x_[i, :], xj)
        return np.where(total >= 0.0, +1, -1)

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

        return self.c1 + self.c2 * kern  

    def predict_final(self, x):
        total = x.shape[0]
        preds = []
        x_df = np.copy(x)
        for j in range(total):
            preds.append(self.predict(x_df[j, :]).tolist())
        return preds


class MCP_Dual(object):

    def __init__(self, num_classifiers, unique_labels, max_iters=200, kernel='none', sigma=1.0, theta=0.0, degree=2.0, eta=1.0, c1=0.0, c2=1.0, classifier='ova'):
        self.classifier = classifier
        self.unique_labels = unique_labels
        self.kernel = kernel
        self.degree = degree
        self.sigma = sigma
        self.num_classifiers = num_classifiers
        
        if classifier == 'ova':
            self.tot_binary_classifiers = num_classifiers
            self.classifiers = np.empty(num_classifiers, dtype=object) 
            for i in range(int(num_classifiers)):
                self.classifiers[i] = Perceptron_Dual(max_iters, kernel, sigma, theta, degree, eta, c1, c2)  
        
        elif classifier == 'ovo':
            self.tot_binary_classifiers = int((num_classifiers*(num_classifiers - 1))/2)
            self.classifiers = np.empty(self.tot_binary_classifiers, dtype=object) 
            for i in range(int(self.tot_binary_classifiers)):
                self.classifiers[i] = Perceptron_Dual(max_iters, kernel, sigma, theta, degree, eta, c1, c2)
            i = j = k = 0
            self.classifier_labels = []
            while i < len(self.unique_labels):
                j = i+1
                while j < len(self.unique_labels-1):
                    self.classifier_labels.append(np.unique([self.unique_labels[i],self.unique_labels[j]]))
                    j += 1
                    k += 1
                i += 1
                
        elif classifier == 'ecoc':
            self.tot_binary_classifiers = int(2**(num_classifiers-1)-1)
            self.classifiers = np.empty(self.tot_binary_classifiers, dtype=object) 
            for i in range(int(self.tot_binary_classifiers)):
                self.classifiers[i] = Perceptron_Dual(max_iters, kernel, sigma, theta, degree, eta, c1, c2)
            self.code_book = np.random.rand(self.num_classifiers, self.tot_binary_classifiers)
            # Improve code_book
            self.code_book[self.code_book > 0.5] = 1
            self.code_book[self.code_book != 1] = -1
            self.cls_idx = dict((c, i) for i, c in enumerate(self.unique_labels))

    def fit(self, X, y):
        if self.classifier == 'ova':
            for i in range(int(self.tot_binary_classifiers)):            
                classifier_y = np.copy(y)
                updated_y = np.where(classifier_y == self.unique_labels[i], +1, -1)
                self.classifiers[i].fit(X,updated_y)
        
        elif self.classifier == 'ovo':
            for i in range(len(self.classifier_labels)):
                classifier_y = np.copy(y)
                classifier_x = np.copy(X)
                index_aux = np.where(classifier_y[:, None] == np.array(self.classifier_labels[i]).ravel())[0]
                updated_x = classifier_x[index_aux]
                updated_y = classifier_y[index_aux]
                target_y = np.where(updated_y == self.classifier_labels[i][0], +1, -1)
                self.classifiers[i].fit(updated_x, target_y)
        
        elif self.classifier == 'ecoc':
            update_y = np.array([self.code_book[self.cls_idx[y[i]]] for i in range(X.shape[0])])
            for i in range(int(self.tot_binary_classifiers)):
                self.classifiers[i].fit(X, update_y[:, i])
               
    def predict(self, X):
        if self.classifier == 'ova':
            y = np.empty([X.shape[0], self.tot_binary_classifiers])
            for i in range(int(self.tot_binary_classifiers)):   
                y[:,i] = self.classifiers[i].predict_final(X)
            return self.unique_labels[np.argmax(y, axis=1)]
        
        elif self.classifier == 'ovo':
            y = np.empty([X.shape[0], self.tot_binary_classifiers])
            for i in range(int(self.tot_binary_classifiers)): 
                y[:,i] = self.classifiers[i].predict_final(X)
            predictions, counter = self.ovo_voting_both(self, y)
            return self.unique_labels[predictions]
        
        elif self.classifier == 'ecoc':
           y = np.empty([X.shape[0], self.tot_binary_classifiers])
           for i in range(int(self.tot_binary_classifiers)):
               y[:,i] = self.classifiers[i].predict_final(X)
           predictions = euclidean_distances(y, self.code_book).argmin(axis=1)
           return self.unique_labels[predictions]

    @staticmethod
    def ovo_voting_both(self, decision_ovo):
        predictions = np.zeros(len(decision_ovo))
        class_pos, class_neg = self.ovo_class_combinations(self.num_classifiers)
        counter = np.zeros([len(decision_ovo), self.num_classifiers])
        for p in range(len(decision_ovo)):
            for i in range(len(decision_ovo[p])):
                counter[p, class_pos[i]] += decision_ovo[p,i] 
                counter[p, class_neg[i]] -= decision_ovo[p,i] 
            predictions[p] = np.argmax(counter[p])
        predictions = predictions.astype(np.int64)
        return predictions, counter
        
    @staticmethod
    def ovo_class_combinations(n_classes):
        class_pos = []
        class_neg = []
        for c1 in range(n_classes-1):
            for c2 in range(c1+1,n_classes):
                class_pos.append(c1)
                class_neg.append(c2)
        return class_pos, class_neg
         