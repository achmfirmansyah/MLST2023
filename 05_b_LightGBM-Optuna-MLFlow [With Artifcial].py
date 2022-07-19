#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

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
from mlflow.tracking import MlflowClient


import hydra
from omegaconf import DictConfig,OmegaConf
import logging
log = logging.getLogger(__name__)

#optuna.logging.set_verbosity(optuna.logging.WARNING)

pic_=None
train_x=None 
valid_x=None 
train_y=None 
valid_y=None
conf = OmegaConf.load('config/config.yaml')
tracking_uri_=conf['config']['tracking_uri']
client = MlflowClient()
mlflc = MLflowCallback(tracking_uri=tracking_uri_,nest_trials=True,
                      metric_name=['f1_micro','log_loss','roc_auc_score','cohen_kappa'])

def preparing():
    global pic_,train_x, valid_x,train_y,valid_y, mlflc,tracking_uri_
    with open('ml_output/04_05_modeling/feature_selection/fs.pickle', 'rb') as handle:
        fs_=pickle.load(handle)

    lgbm_params=fs_['lightgbm-w_artificial']
    strategy=lgbm_params['params.strategy']
    feature_used=lgbm_params['params.feature_name'].replace('[','').replace(']','').replace('\'','').replace(' ','').split(',')
    feature_used.append('nama_valid')
    if lgbm_params['params.condition']=='w_outlier_':
        data_sample=pd.read_csv("gs://bps-gcp-bucket/MLST2023/preprocessing/sample_"+
                             str(pic_)+'_'+strategy+".csv",sep=',')[feature_used]
    else:
        data_sample=pd.read_csv("gs://bps-gcp-bucket/MLST2023/preprocessing/sample_"+
                             str(pic_) +"_no_outlier"+'_'+strategy+".csv",sep=',')[feature_used]
    columns_data=data_sample.columns.to_list()
    
    columns_data.remove('nama_valid')
    data_sample=data_sample
    X=data_sample[columns_data]
    y=data_sample[['nama_valid']]    
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.2,stratify=y)

@mlflc.track_in_mlflow()
def lgbm_obj(trial):
    global train_x, valid_x,train_y,valid_y, mlflc
    param = {
        "objective": "MultiClass",
        "n_estimators":trial.suggest_categorical("iterations", [100,200,500,1000]),
        "random_seed":1234,
        "learning_rate":trial.suggest_float("learning_rate",0.001,0.1),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "lambda_l1": trial.suggest_float('l2_leaf_reg', 0.000001, 1.0, log = True),
        "lambda_l2":trial.suggest_float('l2_leaf_reg', 0.000001, 1.0, log = True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.2, 0.95, step=0.1
        ),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.2, 0.95, step=0.1
        ),
    }

    
    lgb_= LGBMClassifier(**param)
    sk_fold=StratifiedKFold(n_splits=5,shuffle=False)
    f1_micro=[]
    log_loss_=[]
    roc_auc_score_=[]
    cohen_kappa_score_=[]
    for train_index, test_index in sk_fold.split(train_x, train_y):
        lgb_.fit(train_x.iloc[train_index,],train_y.iloc[train_index,], eval_set=[(valid_x,valid_y)], 
                 verbose=-1)
        preds = lgb_.predict(valid_x)
        preds_proba_=lgb_.predict_proba(valid_x)
        pred_labels = np.rint(preds)
        f1_micro.append(f1_score(valid_y, pred_labels,average='micro'))
        log_loss_.append(log_loss(valid_y,preds_proba_))
        roc_auc_score_.append(roc_auc_score(valid_y, preds_proba_, average="weighted", multi_class="ovr"))
        cohen_kappa_score_.append(cohen_kappa_score(valid_y, pred_labels))
        
    mlflow.log_param('algorithm',"lightgbm")        
    mlflow.log_param('strategy','with artificial')    
    mlflow.lightgbm.log_model(lgb_, "lgbm_model")
    
    return np.mean(f1_micro), np.mean(log_loss_), np.mean(roc_auc_score_),np.mean(cohen_kappa_score_)

@hydra.main(version_base=None, config_path="config", config_name="config")
def myapp(cfg : DictConfig) -> None:
    global pic_, tracking_uri_,mlflc
    pic_=cfg.config.pic_
    tracking_uri_=cfg.config.tracking_uri
    preparing()
    
    study = optuna.create_study(study_name='Hyper Parameter Tuning',load_if_exists=True, 
                                storage='sqlite:///ml_output/05_trials_hypertuning/tuning_lightgbm_with_artificial.db',
                                directions=['maximize','minimize','maximize','maximize'])
    study.optimize(lgbm_obj, n_trials=50, callbacks=[mlflc],n_jobs=1)

    
if __name__ == '__main__':
    myapp()
