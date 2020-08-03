#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 20:03:07 2020

@author: GrupoA - OML - 19/20
"""

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

np.random.seed(0)

class Perceptron_Primal(object):
   
    def __init__(self, learningrate, max_iterations, classifier):
        self.learningrate = learningrate
        self.max_iterations = max_iterations
        self.classifier = classifier
        
    def fit(self, X, y):
        iterations = 0
        mismatch_flag = 1
        self.w = np.array([0] * len(X[0]))
        self.errors_ = []
        while iterations <= self.max_iterations and mismatch_flag != 0:
            mismatch_flag = 0
            sum_error = 0
            #self.w = np.array([0] * len(X[0]))
            for i in range(len(X)):
                y_hat = np.dot(self.w.T, X[i,:])
                if y_hat >= 0.0:
                    y_hat = 1
                else:
                    y_hat = -1
                if y[i]*y_hat <= 0:
                    sum_error += 1
                    mismatch_flag = 1
                    self.w = self.w + self.learningrate * y[i] * X[i]
            self.errors_.append(sum_error)
            iterations += 1
        self.iterations = iterations
 
    def predict(self, X):
        final_scores = np.array([np.dot(self.w.T, x) for x in X])
        preds = [1 if x >= 0.0 else -1 for x in final_scores]
        return preds


class MCP_Primal(object):

    def __init__(self, num_classifiers, unique_labels, max_iters=200, learningrate=0.1, classifier='ova'):
        self.classifier = classifier
        self.unique_labels = unique_labels
        self.num_classifiers = num_classifiers
        
        if classifier == 'ova':
            self.tot_binary_classifiers = num_classifiers
            self.classifiers = np.empty(num_classifiers, dtype=object) 
            for i in range(int(num_classifiers)):
                self.classifiers[i] = Perceptron_Primal(learningrate, max_iters, classifier)  
        
        elif classifier == 'ovo':
            self.tot_binary_classifiers = int((num_classifiers*(num_classifiers - 1))/2)
            self.classifiers = np.empty(self.tot_binary_classifiers, dtype=object) 
            for i in range(int(self.tot_binary_classifiers)):
                self.classifiers[i] = Perceptron_Primal(learningrate, max_iters, classifier)
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
                self.classifiers[i] = Perceptron_Primal(learningrate, max_iters, classifier)
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
                self.classifiers[i].fit(X, updated_y)
        
        elif self.classifier == 'ovo':
            for i in range(int(self.tot_binary_classifiers)):
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
                y[:,i] = self.classifiers[i].predict(X)
            return self.unique_labels[np.argmax(y, axis=1)]  
        
        elif self.classifier == 'ovo':
            y = np.empty([X.shape[0], self.tot_binary_classifiers])
            for i in range(int(self.tot_binary_classifiers)): 
                y[:,i] = self.classifiers[i].predict(X)
            predictions, counter = self.ovo_voting_both(self, y)
            return self.unique_labels[predictions]
        
        elif self.classifier == 'ecoc':
            y = np.empty([X.shape[0], self.tot_binary_classifiers])
            for i in range(int(self.tot_binary_classifiers)):
                y[:,i] = self.classifiers[i].predict(X)
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
    