#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 20:04:30 2020

@author: GrupoA - OML - 19/20
"""

import tensorflow as tf
import numpy as np
from MCP_Primal import MCP_Primal
from MCP_Dual import MCP_Dual 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import load_digits
import time
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import flask
from flask import Flask, request
import io
from PIL import Image
import base64
from flask_cors import CORS

warnings.filterwarnings('ignore')

tf.random.set_seed(0)
np.random.seed(0)
#for an easy reset backend session state
tf.keras.backend.clear_session()


app = Flask(__name__)
cors = CORS(app, resources={r"/predict*": {"origins": "*"}})


###################################################################################################
################################# Functions #######################################################


def perceptron_primal(x_train, y_train, x_test, y_test, learningrate=0.1, classifier='ova', max_iters=200):
    if classifier == 'ova':
        title = 'OneVsAll'
    elif classifier == 'ovo':
        title = 'OneVsOne'
    elif classifier == 'ecoc':
        title = 'ECOC'
    
    # Number of unique class labels  which is also the number of classifiers we will train 
    unique_labels = np.unique([y_train])
    num_classifiers = unique_labels.size
    
    # Train the model
    model = MCP_Primal(num_classifiers, unique_labels, max_iters=max_iters, learningrate=learningrate, classifier=classifier)
    model.fit(x_train, y_train)
    
    # Run predictions on test data 
    y_predicted = model.predict(x_test)
    
    # Score model
    score = accuracy_score(y_test, y_predicted)

    return score, y_predicted, model, title


def perceptron_dual(x_train, y_train, x_test, y_test, classifier='ova', max_iters=200, kernel='none', degree=2):
    if classifier == 'ova':
        title = 'OneVsAll'
    elif classifier == 'ovo':
        title = 'OneVsOne'
    elif classifier == 'ecoc':
        title = 'ECOC'
    
    # Number of unique class labels  which is also the number of classifiers we will train 
    unique_labels = np.unique([y_train])
    num_classifiers = unique_labels.size
    
    # Train the model
    model = MCP_Dual(num_classifiers, unique_labels, max_iters=max_iters, kernel=kernel, classifier=classifier, degree=degree)
    model.fit(x_train, y_train)
    
    # Run predictions on test data 
    y_predicted = model.predict(x_test)
    
    # Score model
    score = accuracy_score(y_test, y_predicted)

    return score, y_predicted, model, title


def confusion_matrix(y_test, predictions, score, title):
    unique_labels = np.unique([y_test])
    cm = metrics.confusion_matrix(y_test, predictions)
    plt.figure(figsize=(12,12))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r', xticklabels=unique_labels, yticklabels=unique_labels);
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = title + ' - Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size=15);
    plt.savefig('temp/confusion_matrix.png')
    plt.close()    
    with open("temp/confusion_matrix.png", "rb") as imageFile:
        img = base64.b64encode(imageFile.read())
    return img


def prepare_data(X, y, lista):    
    threshold = np.where(y[:, None] == np.array(lista).ravel())[0]
    y_new, x_new = y[threshold], X[threshold]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_new_scaled = scaler.fit_transform(x_new)
    x_train, x_test, y_train, y_test = train_test_split(x_new_scaled, y_new, test_size=0.20, random_state=0)
    return x_train, y_train, x_test, y_test


def plot_erros(model, title):
    errors = []
    i = 1
    for classifier in model.classifiers:
        plt.plot(range(1, len(classifier.errors_) + 1), classifier.errors_, marker='o')
        plt.title('Misclassification - ' + title + '- Classifier nº' + str(i))
        plt.xlabel('Iterations')
        plt.ylabel('Number of misclassification')
        plt.savefig('temp/errors.png')
        plt.close()    
        i = i+1
        with open("temp/errors.png", "rb") as imageFile:
            img = base64.b64encode(imageFile.read())
            errors.append(str(img))
    return errors

def plot_iterations(model):
    plot_iterations = []
    for classifier in model.classifiers:
        plot_iterations.append(classifier.iterations)
    return plot_iterations
        

def plot_predictions(teste_x, predicted, teste_y, title):
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(title, size=16)
    for i in range(64):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(teste_x.reshape(-1, 8, 8)[i], cmap=plt.cm.binary, interpolation='nearest')
        if predicted[i] == teste_y[i]:
            ax.text(0, 7, str(predicted[i]), color='green')
        else:
            ax.text(0, 7, str(predicted[i]), color='red') 
    plt.savefig('temp/predictions.png')
    plt.close()    
    with open("temp/predictions.png", "rb") as imageFile:
        img = base64.b64encode(imageFile.read())
    return img


def explore(x, y, num_class):
    plt.figure(figsize=(10, 9))
    plt.title('Decision regions')
    pca = PCA(n_components=num_class)
    proj = pca.fit_transform(x)
    plt.scatter(proj[:, 0], proj[:, 1], c=y, cmap="Paired")
    plt.colorbar()
    plt.savefig('temp/explore.png')
    plt.close()
    with open("temp/explore.png", "rb") as imageFile:
        img = base64.b64encode(imageFile.read())
    return img


###################################################################################################
################################# Load Data #######################################################


# Load digits
digits = load_digits()

# Load mnist
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

X_mnist = np.concatenate((x_train, x_test))
y_mnist = np.concatenate((y_train, y_test))

X_mnist_threshold = X_mnist[:2000]
y_mnist__threshold = y_mnist[:2000]
X_mnist_threshold = X_mnist_threshold.reshape(X_mnist_threshold.shape[0], -1)
y_mnist__threshold = y_mnist__threshold.reshape(y_mnist__threshold.shape[0], -1)
y_mnist__threshold = y_mnist__threshold.ravel()


###################################################################################################
################################# Main ############################################################


# request model prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        content = request.json
    
    if content['dataset']:
        dataset = content['dataset']
    if content['perceptron_type']:
        perceptron_type = content['perceptron_type']
    if content['classes']:
        lista = content['classes']
    if content['classifier']:
        classifier = content['classifier']
        
    if dataset == 'digits':
        treino_x, treino_y, teste_x, teste_y = prepare_data(digits.data, digits.target, lista)
    elif dataset == 'mnist':
        treino_x, treino_y, teste_x, teste_y = prepare_data(X_mnist_threshold, y_mnist__threshold, lista)
        
    unique_labels = np.unique([treino_y])
    num_classifiers = unique_labels.size
    explore_image = explore(treino_x, treino_y, num_classifiers)
    
    if perceptron_type == 'primal':
        if content['learning_rate']:
            learning_rate = float(content['learning_rate'])
        if content['max_iterations']:
            max_iterations = int(content['max_iterations'])
            
        start_time = time.time()
                                
        score, predicted, model, title = perceptron_primal(treino_x, treino_y , teste_x, teste_y, learningrate=learning_rate, classifier=classifier, max_iters=max_iterations)
    
        finish_time = time.time() - start_time
        
        errors = plot_erros(model, title)
        
        iterations = plot_iterations(model)
        
        confusion_matrix_image = confusion_matrix(teste_y, predicted, score, title + ' - Primal')
        
        if dataset == 'digits':
            predictions_image = plot_predictions(teste_x, predicted, teste_y, title + ' - Primal - ' + str(score))
            data = {
                'score': score,
                'errors': errors,
                'title': title,
                'time': finish_time,
                'iterations': iterations,
                'image_explore': str(explore_image),
                'predictions_image': str(predictions_image),
                'confusion_matrix_image': str(confusion_matrix_image)
            }
            
        elif dataset== 'mnist':
            data = {
                'score': score,
                'errors': errors,
                'title': title,
                'time': finish_time,
                'iterations': iterations,
                'image_explore': str(explore_image),
                'confusion_matrix_image': str(confusion_matrix_image)
            }

        return flask.jsonify(data)
    
    elif perceptron_type == 'dual':
        if content['max_iterations']:
            max_iterations = int(content['max_iterations'])
        if content['kernel']:
            kernel = content['kernel']
            
        if kernel == 'polynomial':
            degree = int(content['degree'])
        
            start_time = time.time()
                                    
            score, predicted, model, title = perceptron_dual(treino_x, treino_y , teste_x, teste_y, classifier=classifier, max_iters=max_iterations, kernel=kernel, degree=degree)
        
            finish_time = time.time() - start_time
            
            errors = plot_erros(model, title)
            
            iterations = plot_iterations(model)
        
            confusion_matrix_image = confusion_matrix(teste_y, predicted, score, title + ' - Dual - ' + kernel)
        
        else:
            start_time = time.time()
                                    
            score, predicted, model, title = perceptron_dual(treino_x, treino_y , teste_x, teste_y, classifier=classifier, max_iters=max_iterations, kernel=kernel)
        
            finish_time = time.time() - start_time
            
            errors = plot_erros(model, title)
            
            iterations = plot_iterations(model)
        
            confusion_matrix_image = confusion_matrix(teste_y, predicted, score, title + ' - Dual - ' + kernel)
            
        if dataset == 'digits':
            predictions_image = plot_predictions(teste_x, predicted, teste_y, title + ' - Dual - ' + kernel + ' - ' + str(score))
            data = {
                'score': score,
                'errors': errors,
                'title': title,
                'time': finish_time,
                'iterations': iterations,
                'image_explore': str(explore_image),
                'predictions_image': str(predictions_image),
                'confusion_matrix_image': str(confusion_matrix_image)
            }
            
        elif dataset== 'mnist':
            data = {
                'score': score,
                'errors': errors,
                'title': title,
                'time': finish_time,
                'iterations': iterations,
                'image_explore': str(explore_image),
                'confusion_matrix_image': str(confusion_matrix_image)
            }
    
        return flask.jsonify(data)

# start Flask server
if __name__ == "__main__":
        from waitress import serve
        serve(app, host='0.0.0.0', port=4000)
