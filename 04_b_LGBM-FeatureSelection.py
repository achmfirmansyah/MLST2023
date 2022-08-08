#!/usr/bin/env python
# coding: utf-8

import pandas as pd 
import numpy as np
import os
import scipy
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score,log_loss
import os

from google.cloud import storage
client = storage.Client()
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'config/gcp_store.json'
client = storage.Client()

from multiprocessing import Pool as pool_x

import mlflow
from mlflow.tracking import MlflowClient

import hydra
from omegaconf import DictConfig,OmegaConf
import logging
log = logging.getLogger(__name__)

pic_=None
tracking_uri_=None
status_=None 
train_x=None 
valid_x=None
train_y=None
valid_y=None 
columns_data=None
strategy_=None
columns_data=None

def parallel_FE_lgbm(i):
    global pic_,status_, train_x, valid_x, train_y, valid_y, columns_data, strategy_,tracking_uri_
    
    experiment_name = "Feature Selection"
    ## check if the experiment already exists
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(name=experiment_name) 
    experiment = mlflow.get_experiment_by_name(experiment_name)
    ### check finished experiment
    client = MlflowClient()
    exp_r = client.get_experiment_by_name("Feature Selection")
    client_run=mlflow.search_runs([exp_r.experiment_id], order_by=["metrics.f1_micro DESC"])
    if ((client_run.shape[0]>0)&("params.condition" in client_run.columns)&
        ("params.algorithm" in client_run.columns)&("params.strategy" in client_run.columns)):
        client_run=client_run.loc[(client_run["params.condition"]==status_)&
                                  (client_run["params.algorithm"]=='lightgbm')&
                                  (client_run["params.strategy"]==strategy_)]
        if str(i) in client_run['params.num_feature'].to_list():
            print('Experiment with number of feature:',i, ' already done. It will be skipped.')
            return ''
    mlflow.set_tracking_uri(tracking_uri_)
    
    with mlflow.start_run(experiment_id = experiment.experiment_id,
                          run_name=status_+'-'+datetime.now().strftime("%d/%m/%Y %H:%M:%S")):
        model = lgb.LGBMClassifier(random_state=1234,verbose=-1, n_estimators=50)
        model.fit(train_x, train_y,
                     eval_set=[(valid_x,valid_y)])
        selector = RFE(model, n_features_to_select=i, step=1)
        selector.fit(train_x,train_y)
        
        
        col_data=[columns_data[j] for j in list(np.where(selector.support_)[0])]    
            
        pred_labels=selector.predict(valid_x).tolist()
        preds_proba_=selector.predict_proba(valid_x)
        
        
        f1_micro=f1_score(valid_y, pred_labels,average='micro')
        log_loss_=log_loss(valid_y,preds_proba_)
        
        mlflow.log_param("num_feature", i)
        mlflow.log_param('feature_name',col_data)
        mlflow.log_param('condition',status_)
        mlflow.log_param('algorithm',"lightgbm")        
        mlflow.log_param('strategy',strategy_)

        mlflow.log_metric("f1_micro", f1_micro)
        mlflow.log_metric("log_loss", log_loss_)
        
    return ''

def run_FE_lgbm():
    y1=range(5,18)
    for i1 in y1:
        parallel_FE_lgbm(i1)

def run_noartificial():
    global pic_, status_,strategy_,train_x, valid_x, train_y, valid_y,columns_data
    log.info('RUN DATA WITHOUT ARTIFICIAL DATA')
    log.info('===========RUN Data w Outlier===========')
    data_sample=pd.read_csv("gs://bps-gcp-bucket/MLST2023/preprocessing/sample_"+str(pic_) +".csv",sep=',')
    data_sample['nama_valid']=data_sample.nama_valid.apply(lambda y: str(y)[:6])
    data_sample=data_sample.sort_values('nama_valid')
    status_='w_outlier_'
    columns_data=['B1_p15', 'B2_p15', 'B3_p15',
       'B4_p15', 'B5_p15', 'B6_p15', 'B7_p15', 'B8_p15', 'B8A_p15', 'B11_p15',
       'B12_p15', 'NDVI_p50', 'NDWI_p50', 'NDBI_p50', 'SAVI_p50', 'EVI_p50',
       'GNDVI_p50']
    X=data_sample[columns_data]
    y=data_sample['nama_valid']
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.2,stratify=y)
    strategy_='no artificial'
    run_FE_lgbm()
    log.info('===========FINISH=======================')
    log.info('===========RUN Data w/o Outlier===========')
    data_sample=pd.read_csv("gs://bps-gcp-bucket/MLST2023/preprocessing/sample_"+str(pic_) +"_no_outlier.csv",sep=',')
    data_sample['nama_valid']=data_sample.nama_valid.apply(lambda y: str(y)[:6])
    data_sample=data_sample.sort_values('nama_valid')
    status_='wo_outlier_'
    columns_data=['B1_p15', 'B2_p15', 'B3_p15',
       'B4_p15', 'B5_p15', 'B6_p15', 'B7_p15', 'B8_p15', 'B8A_p15', 'B11_p15',
       'B12_p15', 'NDVI_p50', 'NDWI_p50', 'NDBI_p50', 'SAVI_p50', 'EVI_p50',
       'GNDVI_p50']
    X=data_sample[columns_data]
    y=data_sample['nama_valid']
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.2,stratify=y)
    strategy_='no artificial'
    run_FE_lgbm()
    log.info('===========FINISH=======================') 
    log.info('FINISH')


def run_enn_border_smote():
    global pic_, status_,strategy_,train_x, valid_x, train_y, valid_y,columns_data
    log.info('RUN DATA WITH ENN-BORDER SMOTE')
    log.info('===========RUN Data w Outlier===========')
    data_sample=pd.read_csv("gs://bps-gcp-bucket/MLST2023/preprocessing/sample_"+str(pic_) +"_enn_border_smote.csv",sep=',')
    data_sample['nama_valid']=data_sample.nama_valid.apply(lambda y: str(y)[:6])
    data_sample=data_sample.sort_values('nama_valid')
    status_='w_outlier_'
    columns_data=['B1_p15', 'B2_p15', 'B3_p15',
       'B4_p15', 'B5_p15', 'B6_p15', 'B7_p15', 'B8_p15', 'B8A_p15', 'B11_p15',
       'B12_p15', 'NDVI_p50', 'NDWI_p50', 'NDBI_p50', 'SAVI_p50', 'EVI_p50',
       'GNDVI_p50']
    X=data_sample[columns_data]
    y=data_sample['nama_valid']
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.2,stratify=y)
    strategy_='enn_border_smote'
    run_FE_lgbm()
    log.info('===========FINISH=======================')
    log.info('===========RUN Data w/o Outlier===========')
    data_sample=pd.read_csv("gs://bps-gcp-bucket/MLST2023/preprocessing/sample_"+str(pic_) +"_no_outlier_enn_border_smote.csv",sep=',')
    data_sample['nama_valid']=data_sample.nama_valid.apply(lambda y: str(y)[:6])
    data_sample=data_sample.sort_values('nama_valid')
    status_='wo_outlier_'
    columns_data=['B1_p15', 'B2_p15', 'B3_p15',
       'B4_p15', 'B5_p15', 'B6_p15', 'B7_p15', 'B8_p15', 'B8A_p15', 'B11_p15',
       'B12_p15', 'NDVI_p50', 'NDWI_p50', 'NDBI_p50', 'SAVI_p50', 'EVI_p50',
       'GNDVI_p50']
    X=data_sample[columns_data]
    y=data_sample['nama_valid']
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.2,stratify=y)
    strategy_='enn_border_smote'
    run_FE_lgbm()
    log.info('===========FINISH=======================')
    log.info('FINISH')

def run_enn_smote():
    global pic_, status_,strategy_,train_x, valid_x, train_y, valid_y,columns_data
    log.info('RUN DATA WITH ENN-SMOTE')
    log.info('===========RUN Data w Outlier===========')
    data_sample=pd.read_csv("gs://bps-gcp-bucket/MLST2023/preprocessing/sample_"+str(pic_) +"_enn_smote.csv",sep=',')
    data_sample['nama_valid']=data_sample.nama_valid.apply(lambda y: str(y)[:6])
    data_sample=data_sample.sort_values('nama_valid')
    status_='w_outlier_'
    columns_data=['B1_p15', 'B2_p15', 'B3_p15',
       'B4_p15', 'B5_p15', 'B6_p15', 'B7_p15', 'B8_p15', 'B8A_p15', 'B11_p15',
       'B12_p15', 'NDVI_p50', 'NDWI_p50', 'NDBI_p50', 'SAVI_p50', 'EVI_p50',
       'GNDVI_p50']
    X=data_sample[columns_data]
    y=data_sample['nama_valid']
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.2,stratify=y)
    strategy_='enn_smote'
    run_FE_lgbm()
    log.info('===========FINISH=======================')
    log.info('===========RUN Data w/o Outlier===========')
    data_sample=pd.read_csv("gs://bps-gcp-bucket/MLST2023/preprocessing/sample_"+str(pic_) +"_no_outlier_enn_smote.csv",sep=',')
    data_sample['nama_valid']=data_sample.nama_valid.apply(lambda y: str(y)[:6])
    data_sample=data_sample.sort_values('nama_valid')
    status_='wo_outlier_'
    columns_data=['B1_p15', 'B2_p15', 'B3_p15',
       'B4_p15', 'B5_p15', 'B6_p15', 'B7_p15', 'B8_p15', 'B8A_p15', 'B11_p15',
       'B12_p15', 'NDVI_p50', 'NDWI_p50', 'NDBI_p50', 'SAVI_p50', 'EVI_p50',
       'GNDVI_p50']
    X=data_sample[columns_data]
    y=data_sample['nama_valid']
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.2,stratify=y)
    strategy_='enn_smote'
    run_FE_lgbm()
    log.info('===========FINISH=======================')
    log.info('FINISH')

def run_enn_border_smote():
    global pic_, status_,strategy_,train_x, valid_x, train_y, valid_y,columns_data
    log.info('RUN DATA WITH ENN-BORDER SMOTE')
    log.info('===========RUN Data w Outlier===========')
    data_sample=pd.read_csv("gs://bps-gcp-bucket/MLST2023/preprocessing/sample_"+str(pic_) +"_enn_border_smote.csv",sep=',')
    data_sample['nama_valid']=data_sample.nama_valid.apply(lambda y: str(y)[:6])
    data_sample=data_sample.sort_values('nama_valid')
    status_='w_outlier_'
    columns_data=['B1_p15', 'B2_p15', 'B3_p15',
       'B4_p15', 'B5_p15', 'B6_p15', 'B7_p15', 'B8_p15', 'B8A_p15', 'B11_p15',
       'B12_p15', 'NDVI_p50', 'NDWI_p50', 'NDBI_p50', 'SAVI_p50', 'EVI_p50',
       'GNDVI_p50']
    X=data_sample[columns_data]
    y=data_sample['nama_valid']
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.2,stratify=y)
    strategy_='enn_border_smote'
    run_FE_lgbm()
    log.info('===========FINISH=======================')
    log.info('===========RUN Data w/o Outlier===========')
    data_sample=pd.read_csv("gs://bps-gcp-bucket/MLST2023/preprocessing/sample_"+str(pic_) +"_no_outlier_enn_border_smote.csv",sep=',')
    data_sample['nama_valid']=data_sample.nama_valid.apply(lambda y: str(y)[:6])
    data_sample=data_sample.sort_values('nama_valid')
    status_='wo_outlier_'
    columns_data=['B1_p15', 'B2_p15', 'B3_p15',
       'B4_p15', 'B5_p15', 'B6_p15', 'B7_p15', 'B8_p15', 'B8A_p15', 'B11_p15',
       'B12_p15', 'NDVI_p50', 'NDWI_p50', 'NDBI_p50', 'SAVI_p50', 'EVI_p50',
       'GNDVI_p50']
    X=data_sample[columns_data]
    y=data_sample['nama_valid']
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.2,stratify=y)
    strategy_='enn_border_smote'
    run_FE_lgbm()
    log.info('===========FINISH=======================')
    log.info('FINISH')

def run_tl_smote():
    global pic_, status_,strategy_,train_x, valid_x, train_y, valid_y,columns_data
    log.info('RUN DATA WITH TL-SMOTE')
    log.info('===========RUN Data w Outlier===========')
    data_sample=pd.read_csv("gs://bps-gcp-bucket/MLST2023/preprocessing/sample_"+str(pic_) +"_tl_smote.csv",sep=',')
    data_sample['nama_valid']=data_sample.nama_valid.apply(lambda y: str(y)[:6])
    data_sample=data_sample.sort_values('nama_valid')
    status_='w_outlier_'
    columns_data=['B1_p15', 'B2_p15', 'B3_p15',
       'B4_p15', 'B5_p15', 'B6_p15', 'B7_p15', 'B8_p15', 'B8A_p15', 'B11_p15',
       'B12_p15', 'NDVI_p50', 'NDWI_p50', 'NDBI_p50', 'SAVI_p50', 'EVI_p50',
       'GNDVI_p50']
    X=data_sample[columns_data]
    y=data_sample['nama_valid']
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.2,stratify=y)
    strategy_='tl_smote'
    run_FE_lgbm()
    log.info('===========FINISH=======================')
    log.info('===========RUN Data w/o Outlier===========')
    data_sample=pd.read_csv("gs://bps-gcp-bucket/MLST2023/preprocessing/sample_"+str(pic_) +"_no_outlier_tl_smote.csv",sep=',')
    data_sample['nama_valid']=data_sample.nama_valid.apply(lambda y: str(y)[:6])
    data_sample=data_sample.sort_values('nama_valid')
    status_='wo_outlier_'
    columns_data=['B1_p15', 'B2_p15', 'B3_p15',
       'B4_p15', 'B5_p15', 'B6_p15', 'B7_p15', 'B8_p15', 'B8A_p15', 'B11_p15',
       'B12_p15', 'NDVI_p50', 'NDWI_p50', 'NDBI_p50', 'SAVI_p50', 'EVI_p50',
       'GNDVI_p50']
    X=data_sample[columns_data]
    y=data_sample['nama_valid']
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.2,stratify=y)
    strategy_='tl_smote'
    run_FE_lgbm()
    log.info('===========FINISH=======================')
    log.info('FINISH')

def run_tl_border_smote():
    global pic_, status_,strategy_,train_x, valid_x, train_y, valid_y,columns_data
    log.info('RUN DATA WITH ENN-BORDER SMOTE')
    log.info('===========RUN Data w Outlier===========')
    data_sample=pd.read_csv("gs://bps-gcp-bucket/MLST2023/preprocessing/sample_"+str(pic_) +"_tl_border_smote.csv",sep=',')
    data_sample['nama_valid']=data_sample.nama_valid.apply(lambda y: str(y)[:6])
    data_sample=data_sample.sort_values('nama_valid')
    status_='w_outlier_'
    columns_data=['B1_p15', 'B2_p15', 'B3_p15',
       'B4_p15', 'B5_p15', 'B6_p15', 'B7_p15', 'B8_p15', 'B8A_p15', 'B11_p15',
       'B12_p15', 'NDVI_p50', 'NDWI_p50', 'NDBI_p50', 'SAVI_p50', 'EVI_p50',
       'GNDVI_p50']
    X=data_sample[columns_data]
    y=data_sample['nama_valid']
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.2,stratify=y)
    strategy_='tl_border_smote'
    run_FE_lgbm()
    log.info('===========FINISH=======================')
    log.info('===========RUN Data w/o Outlier===========')
    data_sample=pd.read_csv("gs://bps-gcp-bucket/MLST2023/preprocessing/sample_"+str(pic_) +"_no_outlier_tl_border_smote.csv",sep=',')
    data_sample['nama_valid']=data_sample.nama_valid.apply(lambda y: str(y)[:6])
    data_sample=data_sample.sort_values('nama_valid')
    status_='wo_outlier_'
    columns_data=['B1_p15', 'B2_p15', 'B3_p15',
       'B4_p15', 'B5_p15', 'B6_p15', 'B7_p15', 'B8_p15', 'B8A_p15', 'B11_p15',
       'B12_p15', 'NDVI_p50', 'NDWI_p50', 'NDBI_p50', 'SAVI_p50', 'EVI_p50',
       'GNDVI_p50']
    X=data_sample[columns_data]
    y=data_sample['nama_valid']
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.2,stratify=y)
    strategy_='tl_border_smote'
    run_FE_lgbm()
    log.info('===========FINISH=======================')
    log.info('FINISH')
    

@hydra.main(version_base=None, config_path="config", config_name="config")
def myapp(cfg : DictConfig) -> None:
    global pic_, tracking_uri_,status_,strategy_
    pic_=cfg.config.pic_
    tracking_uri_=cfg.config.tracking_uri
    run_noartificial()
    run_enn_smote()
    run_enn_border_smote()
    run_tl_smote()
    run_tl_border_smote()
    
if __name__=="__main__":
    myapp()