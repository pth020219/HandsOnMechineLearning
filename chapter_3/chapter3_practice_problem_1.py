#!/usr/bin/env python
# coding: utf-8

# In[1]:


# MNIST 데이터셋으로 분류기를 만들어 테스트 세트에서 97% 정확도를 달성해보세요.


# In[50]:


# 라이브러리 import

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


# In[32]:


# mnist 데이터셋 획득

def get_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"]
    y = y.astype(np.uint8)
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    return X_train, X_test, y_train, y_test


# In[34]:


X_train, X_test, y_train, y_test = get_mnist()


# In[46]:


# X data standardization or Normalization

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


# In[47]:


# Pipeline 생성

pipeline = Pipeline([
    ('standard_or_normal', standard_or_normal(select_standard=True)),
    ('knn_clf', KNeighborsClassifier())
])


# In[54]:


# 최적의 KNeighborsClassifier 랜덤 탐색

param = {
    'standard_or_normal__select_standard':[True, False],
    'knn_clf__n_neighbors':range(1, 11),
    'knn_clf__weights':['uniform', 'distance']
}

rs = GridSearchCV(pipeline, param_grid=param, cv=3, verbose=3)
rs.fit(X_train, y_train)


# In[55]:


rs.best_params_


# In[56]:


final_model = rs.best_estimator_

final_prediction = final_model.predict(X_test)

accuracy = accuracy_score(y_test, final_prediction)
print(accuracy)


# In[ ]:




