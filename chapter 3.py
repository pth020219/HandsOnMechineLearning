#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.keys()


# In[2]:


X, y = mnist["data"], mnist["target"]

print(X.shape)

y.shape


# In[3]:


import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap="binary")
plt.axis('off')
plt.show()


# In[4]:


y[0]


# In[5]:


y = y.astype(np.uint8)


# In[6]:


X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


# In[7]:


y_train_5 = (y_train == 5) # 5는 True고, 다른 숫자는 모두 False
y_test_5 = (y_test == 5)


# In[8]:


from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)


# In[9]:


sgd_clf.predict([some_digit])


# In[10]:


from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]
    
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))


# In[11]:


from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")


# In[12]:


from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        return self
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool) 


# In[13]:


never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")


# In[14]:


from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)


# In[15]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)


# In[16]:


y_train_prefect_predictions = y_train_5


# In[17]:


confusion_matrix(y_train_5, y_train_prefect_predictions)


# In[18]:


from sklearn.metrics import precision_score, recall_score

print(precision_score(y_train_5, y_train_pred))
print(recall_score(y_train_5, y_train_pred))


# In[19]:


from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)


# In[20]:


y_scores = sgd_clf.decision_function([some_digit])
y_scores


# In[21]:


threshold = 0
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred


# In[22]:


threshold = 8000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred


# In[23]:


y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")


# In[24]:


from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


# In[25]:


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="precisions")
    plt.plot(thresholds, recalls[:-1], "g-", label="recalls")
    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()


# In[26]:


from sklearn.metrics import PrecisionRecallDisplay

PrecisionRecallDisplay.from_predictions(y_train_5, y_train_pred)

plt.show()


# In[27]:


threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]


# In[28]:


y_train_pred_90 = (y_scores >= threshold_90_precision)


# In[29]:


precision_score(y_train_5, y_train_pred_90)


# In[30]:


recall_score(y_train_5, y_train_pred_90)


# In[31]:


from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


# In[32]:


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')

plot_roc_curve(fpr, tpr)
plt.show()


# In[33]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)


# In[34]:


from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
    
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method='predict_proba')


# In[35]:


y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)


# In[36]:


plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "RandomForest")
plt.legend(loc="lower right")
plt.show()


# In[37]:


roc_auc_score(y_train_5, y_scores_forest)


# In[38]:


from sklearn.svm import SVC

svm_clf = SVC()
svm_clf.fit(X_train, y_train)
svm_clf.predict([some_digit])


# In[39]:


some_digit_scores = svm_clf.decision_function([some_digit])
print(some_digit_scores)


# In[40]:


np.argmax(some_digit_scores)


# In[41]:


svm_clf.classes_


# In[42]:


# SVC 기반의 OVR를 사용하는 다중 분류기

from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SVC())
ovr_clf.fit(X_train, y_train)
print(ovr_clf.predict([some_digit]))
print(len(ovr_clf.estimators_))


# In[43]:


sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])


# In[44]:


sgd_clf.decision_function([some_digit])


# In[45]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")


# In[46]:


y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx


# In[49]:


from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(conf_mx)
disp.plot()
plt.show()


# In[61]:


row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums  * 100 # 백분율로 표시


# In[62]:


np.fill_diagonal(norm_conf_mx, 0)
plt.rc('font', size = 6)
disp = ConfusionMatrixDisplay(norm_conf_mx)
disp.plot()
plt.show()


# In[63]:


from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)


# In[65]:


knn_clf.predict([some_digit])


# In[66]:


y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")


# In[69]:


noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise

noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise

y_train_mod = X_train
y_test_mod = X_test


# In[70]:


knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[0]])

clean_digit_image = clean_digit.reshape(28, 28)

plt.imshow(clean_digit_image, cmap="binary")
plt.axis('off')
plt.show()


# In[ ]:




