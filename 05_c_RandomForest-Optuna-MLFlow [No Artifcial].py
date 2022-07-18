#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pic_='F'
tracking_uri_='http://34.128.104.38:5000'



# In[ ]:


import pandas as pd
import numpy as np


from datetime import datetime

from sklearn.metrics import accuracy_score, f1_score,cohen_kappa_score,roc_auc_score,log_loss
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import pickle5 as pickle
from google.cloud import storage

import os
import matplotlib.pyplot as plt
import seaborn as sns

import optuna
from optuna.integration.mlflow import MLflowCallback
import mlflow
from sklearn.ensemble import RandomForestClassifier
#optuna.logging.set_verbosity(optuna.logging.WARNING)



# In[ ]:


with open('model/feature_selection/fs.pickle', 'rb') as handle:
    fs_=pickle.load(handle)


# In[ ]:


rf_params=fs_['randomforest-wo_artificial']


# In[ ]:


rf_params


# In[ ]:


feature_used=rf_params['params.feature_name'].replace('[','').replace(']','').replace('\'','').replace(' ','').split(',')
feature_used.append('nama_valid')
if rf_params['params.condition']=='w_outlier_':
    data_sample=pd.read_csv("gs://bps-gcp-bucket/MLST2023/preprocessing/sample_"+str(pic_) +".csv",sep=',')[feature_used]
else:
    data_sample=pd.read_csv("gs://bps-gcp-bucket/MLST2023/preprocessing/sample_"+str(pic_) +"_no_outlier.csv",sep=',')[feature_used]


# In[ ]:


columns_data=data_sample.columns.to_list()
columns_data.remove('nama_valid')
data_sample=data_sample
X=data_sample[columns_data]
y=data_sample[['nama_valid']]
    
train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.2,stratify=y)


# In[ ]:


mlflc = MLflowCallback(tracking_uri=tracking_uri_,nest_trials=True,
                      metric_name=['f1_micro','log_loss','roc_auc_score','cohen_kappa'])
@mlflc.track_in_mlflow()
def rf_obj(trial):
    params = {
           "n_estimators":trial.suggest_categorical("iterations", [10,20,50,100]),
           "random_state":1234,
           'max_depth': trial.suggest_int('max_depth', 3, 12),
           'min_samples_split': trial.suggest_int('min_samples_split', 1, 150),
           'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 60),
           "criterion":trial.suggest_categorical('criterion',['gini', 'entropy', 'log_loss']),
           "max_leaf_nodes":trial.suggest_int('max_leaf_nodes',3,12),
        "class_weight":trial.suggest_categorical("class_weight",["balanced", "balanced_subsample",None]),
        "bootstrap":trial.suggest_categorical("bootstrap",[True, False])
        }
    
    rf_= RandomForestClassifier(**params)
    sk_fold=StratifiedKFold(n_splits=5,shuffle=False)
    f1_micro=[]
    log_loss_=[]
    roc_auc_score_=[]
    cohen_kappa_score_=[]
    for train_index, test_index in sk_fold.split(train_x, train_y):
        rf_.fit(train_x.iloc[train_index,],train_y.iloc[train_index,])
        preds = rf_.predict(valid_x)
        preds_proba_=rf_.predict_proba(valid_x)
        pred_labels = np.rint(preds)
        f1_micro.append(f1_score(valid_y, pred_labels,average='micro'))
        log_loss_.append(log_loss(valid_y,preds_proba_))
        roc_auc_score_.append(roc_auc_score(valid_y, preds_proba_, average="weighted", multi_class="ovr"))
        cohen_kappa_score_.append(cohen_kappa_score(valid_y, pred_labels))
        
    mlflow.log_param('algorithm',"randomforest")        
    mlflow.log_param('strategy','no artificial')    
    mlflow.sklearn.log_model(rf_, "rf_model")
    
    return np.mean(f1_micro), np.mean(log_loss_), np.mean(roc_auc_score_),np.mean(cohen_kappa_score_)


# In[ ]:


from mlflow.tracking import MlflowClient
client = MlflowClient()

if __name__ == '__main__':
    study = optuna.create_study(study_name='Hyper Parameter Tuning',load_if_exists=True, 
                                storage='sqlite:///tuning_randomforest_no_artificial.db',
                                directions=['maximize','minimize','maximize','maximize'])
    study.optimize(rf_obj, n_trials=50, callbacks=[mlflc],n_jobs=1)

