import os
import sys
uu = [1316874.18349471,-393579.332035696]
size = 1
buf = 30
IMG_WIDTH = 224
from osgeo import gdal
def array_to_raster(array,xmin,ymax,row,col,proj,name,size):
    dst_filename = name
    x_pixels = col
    y_pixels = row
    PIXEL_SIZE = size
    x_min = xmin
    y_max = ymax
    wkt_projection = proj
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(
        dst_filename,
        x_pixels,
        y_pixels,
        1,
        gdal.GDT_Float32, )
    dataset.SetGeoTransform((
        x_min,    # 0
        PIXEL_SIZE,  # 1
        0,                      # 2
        y_max,    # 3
        0,                      # 4
        -PIXEL_SIZE))  
    dataset.SetProjection(wkt_projection)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()  # Write to disk.

import numpy as np
import copy
import glob
import os
name = glob.glob('../result/preds_test_*.npy')
name = [i.split('/')[-1].split('.npy')[0] for i in name]
print(name)
for n in name:
    preds_test_mod = np.load('../result/'+n+'.npy')
    mask = np.load('../data/mask.npy')
    dim = mask.shape
    numr = dim[0]//(IMG_WIDTH - buf*2)
    numc = dim[1]//(IMG_WIDTH - buf*2)
    count = -1
    for i in range(numr):
        for j in range(numc):
            count += 1    
            temp = preds_test_mod[count][buf:-buf,buf:-buf]
            if j == 0:
                rows = temp
            else:
                rows = np.concatenate((rows,temp),axis = 1)
        if i == 0:
            prediction_map = copy.copy(rows)
        else:
            prediction_map = np.concatenate((prediction_map,rows),axis = 0)
    prediction_map = prediction_map[:,:,0]
    prediction_map = prediction_map*mask[:prediction_map.shape[0],:prediction_map.shape[1]]
    # write out the map
    proj_wkt = 'PROJCS["North_America_Albers_Equal_Area_Conic",GEOGCS["GCS_North_American_1983",DATUM["North_American_Datum_1983",SPHEROID["GRS_1980",6378137,298.257222101]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["longitude_of_center",-96],PARAMETER["Standard_Parallel_1",20],PARAMETER["Standard_Parallel_2",60],PARAMETER["latitude_of_center",40],UNIT["Meter",1],AUTHORITY["EPSG","102008"]]'
    nam = '../result/'+n+'.tif'
    array_to_raster(prediction_map,uu[0],uu[1],dim[0],dim[1],proj_wkt,nam,size)
