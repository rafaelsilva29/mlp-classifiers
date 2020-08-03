#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 20:04:30 2020

@author: GrupoA
"""

import numpy as np
from MCP_Primal import MCP_Primal
from MCP_Dual import MCP_Dual 
import time
import matplotlib.pyplot as plt
import warnings
import pandas as pd

warnings.filterwarnings('ignore')


###################################################################################################
################################# Functions #######################################################


def perceptron_primal(X, y, max_iters=200):
    print('\n')
    print('#'*10 + ' Results - Primal ' + '#'*10)
    
    # Train the model
    model = MCP_Primal(1, max_iters, learningrate=1)
    model.fit(X, y)

def perceptron_dual(X, y, max_iters=200, kernel='none'):
    print('\n')
    print('#'*10 + ' Results - Dual - ' + kernel + ' ' + '#'*10)
    
    # Train the model
    model = MCP_Dual(1, kernel=kernel, degree=16)
    model.fit(X, y)
    

###################################################################################################
################################# Main ############################################################

df = pd.read_csv("../datasets/D.csv", header=None)

X, y = df.iloc[:,0:2].values, df.iloc[:,2:3].values

x_aux1 = df.iloc[:,0:1].values
x_aux2 = df.iloc[:,1:2].values

plt.title('Decision regions')
plt.scatter(x_aux1, x_aux2, c=x_aux2, cmap="RdYlGn", s=50, edgecolors="black")
plt.show()

begin = False
while begin == False:
    print("### Perceptron type ###")
    print("1- Primal;")
    print("2- Dual;")   
    print("3- Kernel (RBF);")
    print("4- Polynomial;")
    print("0- exit;")
    kernel = input("Pick an option: ")
    
    if kernel == '0':
        begin = True
    
    elif kernel == '1' or kernel == '2' or kernel == '3' or kernel == '4':
 
        if kernel == '1':
            start_time = time.time()
             
            perceptron_primal(X, y, max_iters=X.shape[0])
        
            finish_time = time.time() - start_time
        
            print('\n')
            print('Time OneVsAll - Primal: %.5fs' % finish_time)
            print('\n')
                    
        else:
            
            if kernel == '2':
                kernel_type = 'none'
            elif kernel == '3':
                kernel_type = 'rbf'
            elif kernel == '4':
                kernel_type = 'polynomial'
                
            start_time = time.time()
            
            perceptron_dual(X, y,  max_iters=X.shape[0], kernel=kernel_type)
        
            finish_time = time.time() - start_time

            print('\n')
            print('Time OneVsAll - Dual - %s: %.5fs' % (kernel_type, finish_time))
            print('\n')
                    
    elif kernel == '0':
        begin = True
                
    else:
        print("Error: Enter a valid number!")
     
print('\n> Bye bye, see you soon...')
