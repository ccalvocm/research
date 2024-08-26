import os
import ee
import pandas as pd
import datetime
import requests
import geemap

def loadFC():
    in_shp = r'/home/carlos/Downloads/Cuencas_BNA/rioMaipo.shp'
    in_fc = geemap.shp_to_ee(in_shp)
    return in_fc

def download(collection):
    # login()

    # 1. authenticate
    ee.Initialize()

    # define los anios
    yrs=[2019,2020,2021]

    for yr in yrs:
    # 2. define python dates
        fechaIni = datetime.datetime(yr, 12, 1)
        fechaFin = datetime.datetime(yr+1, 2, 28)

        # 3. define the collection
        imageCollection = ee.ImageCollection(collection)

        rectangle=loadFC().geometry()

        image=imageCollection.filterBounds(rectangle).map(qa).map(NDVI).filterDate(fechaIni,fechaFin).median().select('NDVI')

        img_name = "L8" + str(fechaIni)+'_'+str(fechaFin)

        geemap.ee_export_image_to_drive(
            image, description='landsat', folder='export',
            region=rectangle, scale=30,
            fileNamePrefix=img_name
        )

def qa(img):
    QA = img.select("QA_PIXEL")
    shadow = QA.bitwiseAnd(8).neq(0)
    cloud = QA.bitwiseAnd(32).neq(0)
    cloudMaskedImg=ee.Image(img).updateMask(shadow.Not()).updateMask(cloud.Not())
    return cloudMaskedImg

def NDVI(image):
    image = image.multiply(2.75e-05).subtract(0.2)
    ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename("NDVI")
    image = image.addBands(ndvi)
    return image.select("NDVI")

if __name__ == '__main__':
    collection = "LANDSAT/LC08/C02/T1_L2"
    download()
