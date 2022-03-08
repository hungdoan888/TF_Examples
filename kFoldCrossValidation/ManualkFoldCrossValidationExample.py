# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 21:13:42 2022

@author: hungd
"""

#%% Importing required libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
 
#%% For Testing

k = 10

#%% Loading the dataset

def loadData():
    data = load_breast_cancer(as_frame = True)
    df = data.frame
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    return X, y

#%% Create Model

def createModel():
    model = LogisticRegression(solver= 'liblinear')
    return model

#%% Get model accuracty

def evaluateModel(model, X, y):
    # Store acc_scores for each fold
    acc_score = []
    
    # Run k fold cross validation
    kf = KFold(n_splits=k, random_state=None)
    for train_index , test_index in kf.split(X):
        X_train , X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train , y_test = y[train_index] , y[test_index]
         
        model.fit(X_train,y_train)
        pred_values = model.predict(X_test)
         
        acc = accuracy_score(pred_values , y_test)
        acc_score.append(acc)
         
    avg_acc_score = sum(acc_score)/k
     
    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))
    
#%% Main

if __name__ == "__main__":
    # Load data
    X, y = loadData()

    # Create logistic regression model
    model = createModel()
    
    # Evaluate Model
    evaluateModel(model, X, y)
    