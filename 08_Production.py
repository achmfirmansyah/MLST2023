from itertools import repeat
import pickle5 as pickle
import pandas as pd
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata, rasterize_points_radial
import rioxarray as rio
import rasterio
import numpy as np
import geopandas as gpd
from multiprocessing import Pool
import os
from rasterio.merge import merge
from itertools import product
from glob import glob
import sys
from omegaconf import DictConfig,OmegaConf
conf = OmegaConf.load('config/config.yaml')
pic_=conf['config']['pic_']
from pathlib import Path
import dill
with open('ml_output/07_final_model/model_final.pickle', 'rb') as handle:
    model_=dill.load(handle)
best_model=model_[0]
best_fs=model_[1]

def parallel_classify_raster(idgrid):
    tif_name='ml_output/08_class_result/result/classified_'+str(idgrid)+'.tif'
    if Path(tif_name).is_file()==False:
        name_image='duatahun_'+str(idgrid)+'_QALPN1_PakKus_sentinel2_10m.tif'
        dt_=glob('ml_output/08_class_result/raster_temp/'+idgrid+'*.tif')
        dt_.sort()
        ls_file=glob('ml_output/08_class_result/temp/classified_'+idgrid+'*.tif')
        if len(dt_)!=81:
            cutting_raster_temp(idgrid,name_image)
            dt_=glob('ml_output/08_class_result/raster_temp/'+idgrid+'*.tif')
            dt_.sort()
        else:
            print('SKIP RASTER TEMP CUTTING')
        if len(ls_file)!=81:
            for dt1 in dt_:
                pool_paralle(dt1)
            ls_file=glob('ml_output/08_class_result/temp/'+idgrid+'*.tif')
        else:
            print('SKIP TEMP CLASSIFICATION')
            
        tif_name='ml_output/08_class_result/result/classified_'+str(idgrid)+'.tif'
        src_files_to_mosaic = []
        for ls in ls_file:
            src = rasterio.open(ls)
            src_files_to_mosaic.append(src)
        mosaic, out_trans = merge(src_files_to_mosaic)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff","height": mosaic.shape[1],"width": mosaic.shape[2],"transform": out_trans})
        with rasterio.open(tif_name, "w", **out_meta) as dest:
            dest.write(mosaic)
            print(tif_name, 'Done')
    else:
        dt_=glob('ml_output/08_class_result/raster_temp/'+idgrid+'*.tif')
        ls_file=glob('ml_output/08_class_result/temp/classified_'+idgrid+'*.tif')    
        if len(dt_)>0:
            for ij in dt_:
                os.remove(ij)
        if len(ls_file)>0:
            for ij in ls_file:
                os.remove(ij)
        print('SKIP CLASSIFICATION')
    

def cutting_raster_temp(idgrid,name_image):
    data_riox=rio.open_rasterio('gs://bps-gcp-bucket/citra-sentinel2/'+pic_+'/'+name_image,
                                masked=True)
    min_x=min(data_riox.x).values+0
    max_x=max(data_riox.x).values+0
    min_y=min(data_riox.y).values+0
    max_y=max(data_riox.y).values+0
    x_=list(np.linspace(min_x,max_x,10))
    y_=list(np.linspace(min_y,max_y,10))
    rangex=range(0,len(x_)-1)
    rangey=range(0,len(y_)-1)
    list_=[]
    for i,j in product(rangex,rangey):
        minx=x_[i]
        miny=y_[j]
        maxx=x_[i+1]
        maxy=y_[j+1]
        geometries = [{
            'type': 'Polygon',
            'coordinates': [[
                [minx, miny],
                [minx, maxy],
                [maxx, maxy],
                [maxx, miny],
                [minx, miny]
            ]]}]
        if ~(Path("ml_output/08_class_result/raster_temp/"+idgrid+"_"+str(i)+"_"+str(j)+".tif").is_file()):
            clipped = data_riox.rio.clip(geometries,all_touched=True)
            clipped.rio.to_raster("ml_output/08_class_result/raster_temp/"+idgrid+"_"+str(i)+"_"+str(j)+".tif", 
                              compress='LZMA', tiled=True, dtype="float64")
            
            
def pool_paralle(ls):
    global best_model, best_fs
    tif_name='classified_'+ls[-15:]
    if ~(Path('ml_output/08_class_result/temp/'+tif_name).is_file()):
        data_riox=rio.open_rasterio(ls)
        temp_=data_riox.to_dataframe(name='value').reset_index().fillna(0)
        matrixData=temp_[['x','y']].drop_duplicates()
        for h in temp_.band.unique():
            t_=temp_.loc[temp_.band==h,['x','y','value']].rename(columns={'value':data_riox.long_name[h-1]})
            matrixData=matrixData.merge(t_,how='left')
        matrixData=gpd.GeoDataFrame(matrixData, crs='epsg:4326',
                                        geometry=gpd.points_from_xy(matrixData.x,matrixData.y)).to_crs('epsg:3857')
        matrixData['prediction']=best_model.predict(matrixData[best_fs])
        matrixData['prediction']=matrixData['prediction'].astype(np.int32)
        geo_grid = make_geocube(
                            vector_data=matrixData,
                            measurements=['prediction'],
                            resolution=(-10, 10),
                            rasterize_function=rasterize_points_griddata)
        geo_grid.prediction.rio.write_nodata(0,inplace=True)
        tif_name='classified_'+ls[-15:]
        geo_grid.rio.to_raster('ml_output/08_class_result/temp/'+tif_name)
    
if __name__=="__main__":
    idgrid=sys.argv[1]
    parallel_classify_raster(idgrid)