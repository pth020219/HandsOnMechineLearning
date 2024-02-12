#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Write a function that can shift an MNIST image in any direction (left, right, up, or down) by one pixel


# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score
from scipy.ndimage import shift


# In[2]:


def get_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"]
    y = y.astype(np.uint8)
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    return X_train.astype(np.int32), X_test.astype(np.int32), y_train, y_test

X_train, X_test, y_train, y_test = get_mnist()


# In[3]:


# scipy.ndimage.shift 를 활용한 함수 생성

def move_image(x, y, direction):
    xtemp, ytemp = x[:60000].copy(), y[:60000].copy()
    if direction == 'up':
        for i, row in enumerate(xtemp):
            moved_array = shift(row.reshape(28, 28), (-1, 0)).reshape(-1)
            xtemp[i] = moved_array
        return np.vstack([x, xtemp]), np.r_[y, ytemp]
    elif direction == 'down':
        for i, row in enumerate(xtemp):
            moved_array = shift(row.reshape(28, 28), (1, 0)).reshape(-1)
            xtemp[i] = moved_array
        return np.vstack([x, xtemp]), np.r_[y, ytemp]
    elif direction == 'left':
        for i, row in enumerate(xtemp):
            moved_array = shift(row.reshape(28, 28), (0, -1)).reshape(-1)
            xtemp[i] = moved_array
        return np.vstack([x, xtemp]), np.r_[y, ytemp]
    else:
        for i, row in enumerate(xtemp):
            moved_array = shift(row.reshape(28, 28), (0, 1)).reshape(-1)
            xtemp[i] = moved_array
        return np.vstack([x, xtemp]), np.r_[y, ytemp]

            


# In[4]:


for i in ['up', 'down', 'left', 'right']:
    X_train, y_train = move_image(X_train, y_train, direction=i)


# In[5]:


# 총 240000개의 데이터 생성. 따라서 훈련 세트는 300000개의 데이터를 가짐.

print(X_train.shape)
print(y_train.shape)


# In[7]:


# 훈련

class standard_or_normal(BaseEstimator):
    def __init__(self, select_standard = True):
        self.select_standard = select_standard
        self.scaler = StandardScaler()
        self.normalizer = MinMaxScaler()
        
    def fit(self, X, y=None):
        if self.select_standard:
            self.scaler.fit(X)
        else:
            self.normalizer.fit(X)
        return self
    
    def transform(self, X):
        if self.select_standard:
            return self.scaler.transform(X)
        else:
            return self.normalizer.transform(X)
        
pipeline = Pipeline([
    ('standard_or_normal', standard_or_normal(select_standard=True)),
    ('knn_clf', KNeighborsClassifier())
])

param = {
    'standard_or_normal__select_standard':[True, False],
    'knn_clf__n_neighbors':range(1, 11),
    'knn_clf__weights':['uniform', 'distance']
}

rs = RandomizedSearchCV(pipeline, param_distributions=param, n_iter=20, cv=3, verbose=3)
rs.fit(X_train, y_train)


# In[8]:


rs.best_params_


# In[9]:


final_model = rs.best_estimator_

final_prediction = final_model.predict(X_test)

accuracy = accuracy_score(y_test, final_prediction)
print(accuracy)


# In[ ]:




