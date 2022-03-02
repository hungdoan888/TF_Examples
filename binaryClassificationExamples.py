# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 19:44:49 2022

@author: hungd
"""

#%% Package imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from planar_utils import plot_decision_boundary
from planar_utils import load_extra_datasets

import tensorflow as tf

from itertools import chain

#%% For Testing

dataSetChoice = "gaussian_quantiles" # noisy_circles, noisy_moons, blobs, gaussian_quantiles
nodesInHiddenLayer = 4
numEpochs = 5000

#%% set a seed so that the results are consistent

np.random.seed(1) 

#%% Get Data

def getData(dataSetChoice):
    # Datasets
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

    # Create Dataset dictionary
    datasets = {"noisy_circles": noisy_circles,
                "noisy_moons": noisy_moons,
                "blobs": blobs,
                "gaussian_quantiles": gaussian_quantiles}
    
    # Choose Dataset
    dataset = datasets[dataSetChoice]
    
    # Create df
    df = pd.DataFrame({'x1': dataset[0].T[0, :], 'x2': dataset[0].T[1, :], 'y': dataset[1]})
    
    # make blobs binary
    if dataSetChoice == "blobs":
        df['y'] = df['y'].apply(lambda x: 1 if x%2 == 1 else 0)
    
    # Visualize the data
    plt.scatter(df['x1'], df['x2'], c=df['y'], s=20, cmap=plt.cm.Spectral)
    return df

#%% Create Model

def createModel(nodesInHiddenLayer):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(nodesInHiddenLayer, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    return model

#%% Compile Model

def compileModel(model):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['binary_accuracy'])
    return model

#%% Fit the Model

def fitModel(model, df, numEpochs):
    model.fit(df[['x1', 'x2']], df[['y']], epochs=numEpochs)
    return model

#%% Evaluate Model

def evaluateModel(model, df):
    test_loss, test_acc = model.evaluate(df[['x1', 'x2']], df[['y']], verbose=2)
    return test_loss, test_acc

#%% Predict

def predictModel(model, df):
    # Get prediction probability
    predProb = model.predict(df[['x1', 'x2']])
    
    # Turn prediction into a binary
    pred = (predProb > .5)
    pred = list(chain.from_iterable(pred))
    
    # Put pred in df and change to 1 or 0
    df['pred'] = pred
    df['pred'] = df['pred'].apply(lambda x: 1 if x else 0)
    
    # Find num of correct
    df['correct'] = df.apply(lambda row: 1 if row['y'] == row['pred'] else 0, axis=1)
    return df

#%% Plot results

def plotResults(model, df, nodesInHiddenLayer):
    # Define X and Y
    X = df[['x1', 'x2']].to_numpy().T
    Y = df[['y']].to_numpy().reshape(1, len(df[['y']]))
    
    # Plot
    plot_decision_boundary(lambda x: model.predict(x), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(nodesInHiddenLayer))

#%% Main

if __name__ == "__main__":
    # Get Data
    df = getData(dataSetChoice)
    
    # Create Model
    model = createModel(nodesInHiddenLayer)
    
    # Compile Model
    model = compileModel(model)
    
    # Fit Model
    model = fitModel(model, df, numEpochs)
    
    # Evaluate Model
    test_loss, test_acc = evaluateModel(model, df)
    print("test_loss:", test_loss)
    print("test_acc:", test_acc)
    
    # Predict with Model
    df = predictModel(model, df)
    
    # Plot Results
    plotResults(model, df, nodesInHiddenLayer)