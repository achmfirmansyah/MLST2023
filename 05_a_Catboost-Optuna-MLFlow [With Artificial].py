#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
from catboost import Pool, CatBoostClassifier

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

    catboost_params=fs_['catboost-w_artificial']
    strategy=catboost_params['params.strategy']
    feature_used=catboost_params['params.feature_name'].replace('[','').replace(']','').replace('\'','').replace(' ','').split(',')
    feature_used.append('nama_valid')
    if catboost_params['params.condition']=='w_outlier_':
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
def catb_obj(trial):
    global train_x, valid_x,train_y,valid_y, mlflc
    param = {
        "objective": "MultiClass",
        "used_ram_limit": "25gb",
        "iterations":trial.suggest_categorical("iterations", [100,200,500,1000]),
        "random_seed":1234,
        "learning_rate":trial.suggest_float("learning_rate",0.0001,0.1),
        "depth": trial.suggest_int("depth", 6, 16),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Plain"]),
        
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 300),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.000001, 1.0, log = True),
        
        "auto_class_weights":trial.suggest_categorical("auto_class_weights",["Balanced","SqrtBalanced"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        )
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    cat_= CatBoostClassifier(**param)
    sk_fold=StratifiedKFold(n_splits=5,shuffle=False)
    f1_micro=[]
    log_loss_=[]
    roc_auc_score_=[]
    cohen_kappa_score_=[]
    for train_index, test_index in sk_fold.split(train_x, train_y):
        pool_train=Pool(train_x.iloc[train_index,],train_y.iloc[train_index,])
        pool_valid=Pool(valid_x,valid_y)
        
        cat_.fit(pool_train, eval_set=pool_valid, verbose=0, early_stopping_rounds=1000)
        preds = cat_.predict(pool_valid)
        preds_proba_=cat_.predict_proba(pool_valid)
        pred_labels = np.rint(preds)
        f1_micro.append(f1_score(valid_y, pred_labels,average='micro'))
        log_loss_.append(log_loss(valid_y,preds_proba_))
        roc_auc_score_.append(roc_auc_score(valid_y, preds_proba_, average="weighted", multi_class="ovr"))
        cohen_kappa_score_.append(cohen_kappa_score(valid_y, pred_labels))
    
    mlflow.log_param('algorithm',"catboost")        
    mlflow.log_param('strategy','with artificial')    
    mlflow.catboost.log_model(cat_, "catboost_model")
    return np.mean(f1_micro), np.mean(log_loss_), np.mean(roc_auc_score_),np.mean(cohen_kappa_score_)

@hydra.main(version_base=None, config_path="config", config_name="config")
def myapp(cfg : DictConfig) -> None:
    global pic_, tracking_uri_,mlflc
    pic_=cfg.config.pic_
    tracking_uri_=cfg.config.tracking_uri
    preparing()
    
    study = optuna.create_study(study_name='Hyper Parameter Tuning',load_if_exists=True, 
                                storage='sqlite:///ml_output/05_trials_hypertuning/tuning_catboost_with_artificial.db',
                                directions=['maximize','minimize','maximize','maximize'])
    study.optimize(catb_obj, n_trials=50, callbacks=[mlflc],n_jobs=1)

    
if __name__ == '__main__':
    myapp()
